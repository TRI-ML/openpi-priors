"""
Local multi-GPU evaluation driver for a trained checkpoint.

Starts a policy server on GPU 0, then spreads eval trials across GPUs 1..N-1.
Each worker evaluates the same task on a disjoint range of trial seeds; the
per-worker results are merged into a single stats.json under
<checkpoint-dir>/evals/<split>/<task>/<timestamp>/.

Usage (3 GPUs: 1 server + 2 eval workers):
    python examples/robocasa/eval_parallel.py \
        --config pi0_robocasa_smoke_test \
        --checkpoint-dir checkpoints/pi0_robocasa_smoke_test/smoke_test/1 \
        --task CloseBlenderLid \
        --num-trials 50 --num-gpus 3
"""

import argparse
import asyncio
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Resolve repo root for absolute paths: this file lives at <repo>/examples/robocasa/.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def wait_for_server(port, timeout=300):
    import websockets

    async def _check():
        try:
            async with websockets.connect(f"ws://0.0.0.0:{port}"):
                return True
        except Exception:
            return False

    start = time.time()
    while time.time() - start < timeout:
        if asyncio.run(_check()):
            return True
        time.sleep(5)
    return False


def _detect_num_egl_devices():
    """Detect number of EGL rendering devices available."""
    try:
        from mujoco.egl import egl_ext as EGL
        return len(EGL.eglQueryDevicesEXT())
    except Exception:
        return 0


def run_eval_worker(env_name, gpu_id, log_dir, split, num_trials, port, seed, trial_offset=0):
    """Launch a single eval worker as a subprocess."""
    env = os.environ.copy()

    # Eval workers need separate GPUs from the server (GPU 0) to avoid BLAS conflicts.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Force osmesa (CPU rendering) for all workers to ensure deterministic ICs.
    # EGL vs osmesa produce different MuJoCo env.reset() behavior, so mixing backends
    # across workers (which happens when num_egl < num_workers) breaks IC determinism.
    # osmesa is slower but guarantees identical results across workers and machines.
    env["MUJOCO_GL"] = "osmesa"
    env["PYOPENGL_PLATFORM"] = "osmesa"
    env.pop("MUJOCO_EGL_DEVICE_ID", None)

    examples_dir = str(REPO_ROOT / "examples" / "robocasa")
    code = f"""
import logging, sys
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, {examples_dir!r})
from main import eval_env
eval_env(
    env_name={env_name!r},
    split={split!r},
    log_dir={log_dir!r},
    num_trials={num_trials},
    resize_size=224,
    replan_steps=5,
    host='0.0.0.0',
    port={port},
    seed={seed},
    trial_offset={trial_offset},
)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def merge_eval_results(checkpoint_dir, task_name, split, worker_dirs):
    """Merge results from multiple workers into a single stats.json.

    Returns (total_episodes, success_rate) or (0, 0.0) if no results.
    """
    total_episodes = 0
    total_successes = 0
    all_rollout_results = []

    for wdir in worker_dirs:
        eval_base = pathlib.Path(wdir) / "evals" / split / task_name
        if not eval_base.exists():
            continue
        for stats_file in eval_base.rglob("stats.json"):
            with open(stats_file) as f:
                stats = json.load(f)
            n = stats.get("num_episodes", 0)
            total_episodes += n
            total_successes += round(stats.get("success_rate", 0) * n)
            all_rollout_results.extend(stats.get("rollout_results", []))

    if total_episodes == 0:
        logging.warning("No eval results found to merge")
        return 0, 0.0

    success_rate = total_successes / total_episodes

    # Write merged stats to the main checkpoint dir
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    merged_dir = pathlib.Path(checkpoint_dir) / "evals" / split / task_name / now
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_stats = {
        "num_episodes": total_episodes,
        "success_rate": success_rate,
        "rollout_results": all_rollout_results,
    }
    with open(merged_dir / "stats.json", "w") as f:
        json.dump(merged_stats, f)

    logging.info(f"Merged: {task_name} — {total_successes}/{total_episodes} = {success_rate:.2%}")

    # Move videos from worker dirs to merged dir
    video_idx = 0
    for wdir in worker_dirs:
        eval_base = pathlib.Path(wdir) / "evals" / split / task_name
        if not eval_base.exists():
            continue
        for mp4 in eval_base.rglob("*.mp4"):
            dest = merged_dir / f"rollout_{video_idx}_{mp4.stem.split('_')[-1]}.mp4"
            mp4.rename(dest)
            video_idx += 1

    # Clean up worker temp dirs
    for wdir in worker_dirs:
        shutil.rmtree(wdir, ignore_errors=True)

    return total_episodes, success_rate


def log_eval_to_wandb(checkpoint_dir, task_name, total_episodes, success_rate):
    """Log eval results to the same wandb run as training.

    No-op unless scripts/train.py wrote <exp_dir>/wandb_id.txt (i.e. training ran with wandb enabled).
    """
    # wandb_id.txt is in the experiment dir (parent of step dir)
    wandb_id_file = pathlib.Path(checkpoint_dir).parent / "wandb_id.txt"
    if not wandb_id_file.exists():
        logging.info("No wandb_id.txt found, skipping wandb logging for eval")
        return

    try:
        import wandb
        run_id = wandb_id_file.read_text().strip()
        project = os.environ.get("WANDB_PROJECT", "openpi")
        entity = os.environ.get("WANDB_ENTITY", None)
        # Use checkpoint step number so eval metrics align with training loss on x-axis
        step = int(pathlib.Path(checkpoint_dir).name)
        wandb.init(id=run_id, resume="must", project=project, entity=entity)
        wandb.log({
            f"eval/{task_name}/success_rate": success_rate,
            f"eval/{task_name}/num_episodes": total_episodes,
        }, step=step)
        wandb.finish()
        logging.info(f"Logged eval results to wandb run {run_id}")
    except Exception as e:
        logging.warning(f"Failed to log eval to wandb: {e}")


def main():
    parser = argparse.ArgumentParser(description="Local multi-GPU parallel eval of a checkpoint")
    parser.add_argument("--config", required=True, help="Training config name")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint step dir")
    parser.add_argument("--task", required=True, help="Task name to evaluate")
    parser.add_argument("--split", default="target", choices=["pretrain", "target"])
    parser.add_argument("--num-trials", type=int, default=50, help="Total rollouts")
    parser.add_argument("--num-gpus", type=int, default=8, help="Total GPUs available (GPU 0 = server, rest = eval workers)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--weights-dir", default=None,
                        help="Load weights from this dir instead of checkpoint-dir (results are still written under checkpoint-dir)")
    args = parser.parse_args()

    # If weights-dir not set, load weights from checkpoint-dir (normal mode)
    if args.weights_dir is None:
        args.weights_dir = args.checkpoint_dir

    num_workers = args.num_gpus - 1  # GPU 0 is for server
    assert num_workers >= 1, "Need at least 2 GPUs (1 server + 1 eval)"

    # Distribute trials across workers
    base_trials = args.num_trials // num_workers
    remainder = args.num_trials % num_workers
    worker_trials = [base_trials + (1 if i < remainder else 0) for i in range(num_workers)]
    worker_trials = [t for t in worker_trials if t > 0]
    num_workers = len(worker_trials)

    # Detect EGL devices once (shared by all workers)
    logging.info(f"Task: {args.task}, {args.num_trials} trials across {num_workers} workers (osmesa rendering for determinism)")
    logging.info(f"Trials per worker: {worker_trials}")

    # Start policy server on GPU 0
    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = "0"
    server_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    serve_script = str(REPO_ROOT / "scripts" / "serve_policy.py")
    logging.info("Starting policy server on GPU 0...")
    server_proc = subprocess.Popen(
        [
            sys.executable, serve_script,
            f"--port={args.port}",
            "policy:checkpoint",
            f"--policy.config={args.config}",
            f"--policy.dir={args.weights_dir}",
        ],
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        logging.info("Waiting for server to load model...")
        if not wait_for_server(args.port, timeout=300):
            logging.error("Server failed to start within timeout")
            # Print server output for debugging
            if server_proc.stdout:
                output = server_proc.stdout.read().decode()[-2000:]
                logging.error(f"Server output: {output}")
            server_proc.kill()
            sys.exit(1)
        logging.info("Server is ready.")

        # Launch workers — each writes to a unique temp log dir.
        # Use a global seed + trial_offset so that trial i always gets seed (global_seed + i)
        # regardless of how trials are distributed across workers.
        global_seed = args.seed
        workers = []
        worker_dirs = []
        trial_offset = 0
        for i in range(num_workers):
            gpu_id = i + 1
            worker_log_dir = f"{args.checkpoint_dir}/.eval_worker_{i}"
            worker_dirs.append(worker_log_dir)

            logging.info(f"  GPU {gpu_id}: {args.task} x{worker_trials[i]} (trials {trial_offset}-{trial_offset + worker_trials[i] - 1})")
            proc = run_eval_worker(
                args.task, gpu_id, worker_log_dir,
                args.split, worker_trials[i], args.port, global_seed,
                trial_offset=trial_offset,
            )
            trial_offset += worker_trials[i]
            workers.append((gpu_id, proc))

        # Wait for all workers
        for gpu_id, proc in workers:
            stdout, _ = proc.communicate()
            if proc.returncode != 0:
                logging.error(f"  GPU {gpu_id} failed (exit {proc.returncode})")
                if stdout:
                    logging.error(f"  Output: {stdout.decode()[-1000:]}")
            else:
                logging.info(f"  GPU {gpu_id} done.")

        # Merge results and log to wandb
        total_episodes, success_rate = merge_eval_results(
            args.checkpoint_dir, args.task, args.split, worker_dirs
        )
        if total_episodes > 0:
            log_eval_to_wandb(args.checkpoint_dir, args.task, total_episodes, success_rate)

        # Print results
        logging.info("=== Evaluation Results ===")
        get_stats_script = str(REPO_ROOT / "examples" / "robocasa" / "get_eval_stats.py")
        subprocess.run(
            [sys.executable, get_stats_script, "--dir", args.checkpoint_dir],
        )

    finally:
        logging.info("Shutting down server...")
        server_proc.kill()
        server_proc.wait()
        logging.info("Done.")


if __name__ == "__main__":
    main()

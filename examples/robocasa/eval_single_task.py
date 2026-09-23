"""
Single-command evaluation: starts the policy server, runs rollouts, reports results, then exits.
Usage:
    python examples/robocasa/eval_single_task.py \
        --config pi0_robocasa_smoke_test \
        --checkpoint-dir checkpoints/pi0_robocasa_smoke_test/smoke_test/9 \
        --env-name CloseBlenderLid \
        --split target \
        --num-trials 2 \
        --server-gpu 0 \
        --eval-gpu 1
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

import requests


def wait_for_server(port, timeout=180):
    """Wait until the websocket server is accepting connections."""
    import websockets
    import asyncio

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


def main():
    parser = argparse.ArgumentParser(description="Single-command eval: server + rollouts + results")
    parser.add_argument("--config", required=True, help="Training config name (e.g. pi0_robocasa_smoke_test)")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint (e.g. checkpoints/.../9)")
    parser.add_argument("--env-name", required=True, help="RoboCasa env name (e.g. CloseBlenderLid)")
    parser.add_argument("--split", default="target", choices=["pretrain", "target"])
    parser.add_argument("--num-trials", type=int, default=2, help="Number of rollout episodes")
    parser.add_argument("--server-gpu", type=int, default=0, help="GPU for policy server")
    parser.add_argument("--eval-gpu", type=int, default=1, help="GPU for eval environment")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Start policy server
    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(args.server_gpu)
    server_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    logging.info(f"Starting policy server on GPU {args.server_gpu}...")
    server_proc = subprocess.Popen(
        [
            sys.executable, "scripts/serve_policy.py",
            f"--port={args.port}",
            "policy:checkpoint",
            f"--policy.config={args.config}",
            f"--policy.dir={args.checkpoint_dir}",
        ],
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # Wait for server to be ready
        logging.info("Waiting for server to load model (this takes ~2 min)...")
        if not wait_for_server(args.port, timeout=300):
            logging.error("Server failed to start within timeout")
            server_proc.kill()
            sys.exit(1)
        logging.info("Server is ready.")

        # Run evaluation
        eval_env = os.environ.copy()
        eval_env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
        eval_env["MUJOCO_GL"] = "egl"

        logging.info(f"Running {args.num_trials} rollouts of {args.env_name} on GPU {args.eval_gpu}...")
        eval_proc = subprocess.run(
            [
                sys.executable, "-c",
                f"""
import logging, sys
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, 'examples/robocasa')
from main import eval_env
eval_env(
    env_name='{args.env_name}',
    split='{args.split}',
    log_dir='{args.checkpoint_dir}',
    num_trials={args.num_trials},
    resize_size=224,
    replan_steps=5,
    host='0.0.0.0',
    port={args.port},
    seed=7,
)
""",
            ],
            env=eval_env,
        )

        if eval_proc.returncode != 0:
            logging.error(f"Evaluation failed with return code {eval_proc.returncode}")
        else:
            logging.info("Evaluation complete.")

        # Report results
        logging.info("Results:")
        subprocess.run(
            [sys.executable, "examples/robocasa/get_eval_stats.py", "--dir", args.checkpoint_dir],
        )

    finally:
        logging.info("Shutting down server...")
        server_proc.kill()
        server_proc.wait()
        logging.info("Done.")


if __name__ == "__main__":
    main()

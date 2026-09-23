import collections
import dataclasses
import logging
import math
import pathlib
import imageio
from datetime import datetime
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json
import os
import robocasa.utils.robomimic.robomimic_dataset_utils as FileUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
import robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
import gymnasium as gym
from robocasa.utils.env_utils import convert_action

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    split: str = "pretrain"
    num_trials: int = 50  # Number of rollouts per task
    task_set: list = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    log_dir: str = None

    seed: int = 7  # Random Seed (for reproducibility)


def eval_main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    split = args.split
    log_dir = args.log_dir
    num_trials = args.num_trials
    resize_size = args.resize_size
    replan_steps = args.replan_steps
    host = args.host
    port = args.port

    all_env_names = []
    for task in args.task_set:
        env_names = TASK_SET_REGISTRY[task]
        all_env_names.extend(env_names)

    for env_name in all_env_names:
        # try:
        eval_env(env_name, split, log_dir, num_trials, resize_size, replan_steps, host, port, args.seed)
        # except Exception as e:
        #     print("Exception!")
        #     print(e)

def eval_env(env_name, split, log_dir, num_trials, resize_size, replan_steps, host, port, seed, trial_offset=0):
    """Evaluate a policy on a RoboCasa task.

    Args:
        trial_offset: Global trial index offset. The environment seed for trial i is
            (seed + trial_offset + i), ensuring identical initial conditions regardless
            of how trials are distributed across workers or how many total trials are run.
    """
    # set args based on task
    assert split in ["pretrain", "target"]
    task_horizon = get_task_horizon(env_name)
    # set dataset path and horizon
    horizon = int(task_horizon * 1.5) # the policy moves slow so give the policy extra time

    now = datetime.now()
    now_formatted = now.strftime("%Y-%m-%d-%H-%M")
    log_path = f"{log_dir}/evals/{split}/{env_name}/{now_formatted}"

    for root, dirs, files in os.walk(os.path.dirname(log_path)):
        if "stats.json" in files:
            print(f"{env_name}/{split}, stats path exists, skipping.")
            return

    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)


    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    rollout_results = []  # per-rollout success labels
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials)):
        # Create a fresh env for each trial with a deterministic seed based on global trial index.
        # This ensures trial i always gets the same initial condition regardless of:
        # - which worker runs it
        # - how many total trials are being run
        # - which policy is being evaluated
        global_trial_idx = trial_offset + episode_idx
        trial_seed = seed + global_trial_idx

        # Seed ALL random sources for full determinism:
        # 1. np.random (global) — some RoboCasa code uses np.random.normal/uniform directly
        #    instead of env.rng (e.g., camera_utils.py eye_in_hand noise, counter.py colors)
        # 2. Python random — used by some sampling code
        # 3. env.rng — RoboCasa's primary rng (set via gym.make seed parameter)
        import random
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        env = gym.make(f"robocasa/{env_name}", split=split, seed=trial_seed)

        # Reset environment
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()

        # Setup
        t = 0
        replay_images = []

        logging.info(f"Starting episode {task_episodes+1}...")
        while t < horizon:
            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
            )

            # Save preprocessed image for replay video
            # replay_images.append(img)

            if not action_plan:
                state = np.concatenate(
                    (
                        obs["state.end_effector_position_relative"],
                        obs["state.end_effector_rotation_relative"],
                        obs["state.base_position"],
                        obs["state.base_rotation"],
                        obs["state.gripper_qpos"],
                    ), axis=0
                )
                # state = np.ascontiguousarray(state)
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": task_lang,
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: replan_steps])

            action = action_plan.popleft()
            action = convert_action(action)
            # Execute action in environment
            obs, reward, done, truncated, info = env.step(action)
            done = info["success"] # for robocasa, usuccess entry in info
            if t % 2 == 0 or t == horizon - 1 or done:
                # Build mosaic from all 3 camera views
                cam_left = image_tools.convert_to_uint8(np.ascontiguousarray(obs["video.robot0_agentview_left"]))
                cam_right = image_tools.convert_to_uint8(np.ascontiguousarray(obs["video.robot0_agentview_right"]))
                cam_wrist = image_tools.convert_to_uint8(np.ascontiguousarray(obs["video.robot0_eye_in_hand"]))
                # Resize wrist cam to match height of the other cameras
                h = cam_left.shape[0]
                cam_wrist_resized = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(cam_wrist, h, h)
                )
                mosaic = np.concatenate([cam_left, cam_right, cam_wrist_resized], axis=1)
                replay_images.append(mosaic)
            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1

        task_episodes += 1
        total_episodes += 1
        rollout_results.append(int(done))

        # Save a replay video of the episode
        suffix = "success" if done else "failure"
        imageio.mimwrite(
            pathlib.Path(log_path) / f"rollout_{global_trial_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=20,
        )

        # Close env (created fresh each trial for deterministic seeding)
        env.env.close()
        del env.env
        del env

        # Log current results
        logging.info(f"Trial {global_trial_idx} (seed={trial_seed}): {'success' if done else 'failure'}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

    logging.info(f"[{env_name}] Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"[{env_name}] Total episodes: {total_episodes}")
    print()
    with open(os.path.join(log_path, "stats.json"), "w") as f:
        stats = {
            "num_episodes": total_episodes,
            "success_rate": float(total_successes) / float(total_episodes),
            "rollout_results": rollout_results,
        }
        json.dump(stats, f, indent=4)




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_main)

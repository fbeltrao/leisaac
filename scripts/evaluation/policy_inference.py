"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="gr00tn1.5",
    help="Type of policy to use. support gr00tn1.5, lerobot-<model_type>, openpi",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")
parser.add_argument("--record", action="store_true", help="Record videos during evaluation.")
parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow logging during evaluation.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time
from os import PathLike
import os
import uuid
import carb
import gymnasium as gym
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)
from leisaac.enhance.envs.metric_logger import MLflowLoggerWrapper

import leisaac  # noqa: F401


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True
    
def resolve_checkpoint(path: PathLike) -> str:
    """
    Resolves the model information based on checkpoint path
    The path can contain sub-folders for multiple epochs. Unless specified, pick the latest epoch.
    The epoch sub-folder might contain sub-folders. Prioritize "pretrained_model" folder if exists.
    The pretrained_model folder might contain a config.json file. If exists, loads parameters from there:
        - "type": is the model type
        - "chunk_size": contains the action horizon used during training
        - 
    """
    pass
    


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "lerobot", "openpi"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""


    # Check if checkpoint path exists
    if not os.path.exists(args_cli.policy_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {args_cli.policy_checkpoint_path} does not exist")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    # modify configuration
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    # create environment
    render_mode = "rgb_array" if args_cli.record else None
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Add recording
    recording_path: PathLike | None = None
    if args_cli.record:
        # Recording uses the Gym RecordVideo wrapper
        # It makes a single recording for the entire evaluation session
        # We provide the output path to save the video files for the MLflowLoggerWrapper,
        # so that the video is avalilable as an artifact in MLflow.
        recording_path = f"./datasets/videos/{uuid.uuid4()}/"
        print(f"[INFO] Recording videos during evaluation in {recording_path}")

        unwrapped_env_for_terminations = env.unwrapped
        def env_had_terminations() -> bool:
            """
            Triggers a new recording if one of the environments had a termination.
            Only works as long as we have a single environment (num_envs=1).
            """
            nonlocal unwrapped_env_for_terminations
            return unwrapped_env_for_terminations.termination_manager.terminated.any() or unwrapped_env_for_terminations.termination_manager.time_outs.any()

        env = gym.wrappers.RecordVideo(env,
                                       video_folder=recording_path,
                                       name_prefix="eval_video",
                                       disable_logger=True,
                                       episode_trigger=lambda episode: True
                                       )

    # Add MLflow logging
    policy_info = {
        "policy_type": args_cli.policy_type,
        "checkpoint_path": args_cli.policy_checkpoint_path,
    }
    

    if not args_cli.disable_mlflow:
        env = MLflowLoggerWrapper(env, 
                                  policy=policy_info,
                                  task=args_cli.task, 
                                  task_description=args_cli.policy_language_instruction,
                                  number_of_episodes=args_cli.eval_rounds,
                                  artifact_paths=[recording_path])

    # create policy
    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00tServicePolicyClient

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.unwrapped.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )
    elif "lerobot" in args_cli.policy_type:
        from isaaclab.sensors import Camera
        from leisaac.policy import LeRobotServicePolicyClient

        model_type = "lerobot"

        policy_type = args_cli.policy_type.split("-")[1]
        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos={
                key: sensor.image_shape for key, sensor in env.unwrapped.scene.sensors.items() if isinstance(sensor, Camera)
            },
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )
    elif args_cli.policy_type == "openpi":
        from isaaclab.sensors import Camera
        from leisaac.policy import OpenPIServicePolicyClient

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[key for key, sensor in env.unwrapped.scene.sensors.items() if isinstance(sensor, Camera)],
            task_type=task_type,
        )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    # reset environment
    obs_dict, _ = env.reset()
    controller.reset()

    # record the results
    success_count, episode_count = 0, 1

    # simulate environment
    unwrapped_env = env.unwrapped
    device = unwrapped_env.device
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    obs_dict, _ = env.reset()
                    episode_count += 1
                    break

                obs_dict = preprocess_obs_dict(obs_dict["policy"], model_type, args_cli.policy_language_instruction)
                actions = policy.get_action(obs_dict).to(device)
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    if unwrapped_env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(unwrapped_env, task_type)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    if reset_terminated[0]:
                        success = True
                        if max_episode_count > 0 and episode_count < max_episode_count:
                            obs_dict, _ = env.reset()
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        if max_episode_count > 0 and episode_count < max_episode_count:
                            obs_dict, _ = env.reset()
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success:
                print(f"[Evaluation] Episode {episode_count} is successful!")
                episode_count += 1
                success_count += 1
                break
            if time_out:
                print(f"[Evaluation] Episode {episode_count} timed out!")
                episode_count += 1
                break
        print(
            f"[Evaluation] now success rate: {success_count / (episode_count - 1)} "
            f" [{success_count}/{episode_count - 1}]"
        )
    print(
        f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
        f" [{success_count}/{max_episode_count}]"
    )

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()

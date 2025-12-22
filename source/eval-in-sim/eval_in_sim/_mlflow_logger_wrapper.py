from __future__ import annotations

import os
import shutil
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, TypedDict

import gymnasium as gym
import mlflow
import pandas as pd
import torch
from isaaclab.envs import ManagerBasedRLEnv

from ._env_utils import detect_termination


class PolicyInfo(TypedDict):
    policy_type: str
    checkpoint_path: os.PathLike | None

class _EvaluationMetricsLogger:
    """
    Logs policy-related metrics and information to MLflow.
    """

    def __init__(
        self,
        policy: PolicyInfo,
        task: str,
        mlflow_tracking_uri: str,
        *,
        params: dict[str, Any] | None = None,
    ):
        self.policy = policy
        self.task = task
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.params = params or {}
        self.params["policy_type"] = policy["policy_type"]
        self.attempts = []
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.failed_attempts_by_reason = {}
        self.successful_attempt_duration = 0
        self.failed_attempt_duration = 0
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.owns_mlflow_run = False

    def _register_attempt(
        self,
        succeeded: bool,
        termination_reason: str = "",
        duration: float = 0.0,
        extra_metrics: dict[str, float] | None = None,
    ):
        extra_metrics = extra_metrics or {}
        self.attempts.append(
            {
                "policy_type": self.policy["policy_type"],
                "task": self.task,
                "recording_id": f"eval_video-episode-{len(self.attempts)}",
                "attempt": len(self.attempts) + 1,
                "succeeded": succeeded,
                "termination_reason": termination_reason,
                "duration": duration,
                **extra_metrics,
            }
        )
        self._log_metrics_episode(
            len(self.attempts), succeeded=succeeded, duration=duration, extra_metrics=extra_metrics
        )

    def _upload_registered_attempts(self) -> None:
        df = pd.DataFrame(self.attempts)

        # Create temporary directory and save CSV there
        temp_dir = tempfile.mkdtemp()
        temp_csv_path = os.path.join(temp_dir, "results.csv")
        df.to_csv(temp_csv_path, index=False)
        mlflow.log_artifact(temp_csv_path)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

    def log_failed_episode(self, termination_reason: str, duration: float, extra_metrics: dict[str, float]) -> None:
        self._register_attempt(
            succeeded=False, termination_reason=termination_reason, duration=duration, extra_metrics=extra_metrics
        )

    def log_successful_episode(self, duration: float, extra_metrics: dict[str, float]) -> None:
        self._register_attempt(succeeded=True, duration=duration, extra_metrics=extra_metrics)

    def _log_metrics_episode(
        self, step: int, succeeded: bool, duration: float, extra_metrics: dict[str, float] | None = None
    ) -> None:
        mlflow.log_metrics(
            {
                "successful_attempts": int(succeeded),
                "failed_attempts": int(not succeeded),
                "duration": duration,
                **(extra_metrics or {}),
            },
            step=step,
        )

    def _log_metrics_aggregated(self) -> None:
        df = pd.DataFrame(self.attempts)
        exclude_columns = [
            "model_version",
            "attempt",
            "succeeded",
            "duration",
        ]
        if df.shape[0] == 0:
            return
        termination_counts = {
            f"count_termination_reason_{reason}": count
            for reason, count in df[df["succeeded"] == False].groupby("termination_reason").size().to_dict().items()
        }
        mlflow.log_metrics(
            {
                "success_rate": df["succeeded"].sum() / df.shape[0],
                "average_success_duration": df[df["succeeded"] == True]["duration"].mean(),
                "average_failure_duration": df[df["succeeded"] == False]["duration"].mean(),
                **{
                    f"average_{col}": df[col].mean()
                    for col in df.columns
                    if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])
                },
                **termination_counts,
            }
        )

    def start(self):
        """Starts the MLflow logging session."""
        is_running_in_azureml = os.getenv("AZUREML_RUN_ID") is not None
        print(f"[INFO] Azure ML detected: {is_running_in_azureml}")
        if not is_running_in_azureml:
            if not os.getenv("MLFLOW_EXPERIMENT_NAME"):
                mlflow.set_experiment(f"{self.task}")
            mlflow.start_run()
            self.owns_mlflow_run = True
        else:
            self.owns_mlflow_run = False

        mlflow.log_params(self.params)
        mlflow.set_tags(
            {
                "task": self.task,
                "job_type": "policy_evaluation_in_sim",
            }
        )

        return self

    def close(self) -> None:
        self._log_metrics_aggregated()
        self._upload_registered_attempts()
        if self.owns_mlflow_run:
            mlflow.end_run()

    def log_artifacts(self, artifact_paths: list[os.PathLike]) -> None:
        """Logs artifacts to MLflow.

        Args:
            artifact_paths (list[os.PathLike]): List of file or directory paths to log as artifacts.
        """
        for path in artifact_paths:
            if path and os.path.exists(path):
                mlflow.log_artifacts(path)

    @staticmethod
    def _get_isaac_versions() -> tuple[str, str]:
        try:
            isaac_sim = version("isaacsim")
        except PackageNotFoundError:
            isaac_sim = "unknown"

        try:
            isaac_lab = version("isaaclab")
        except PackageNotFoundError:
            isaac_lab = "unknown"

        return isaac_sim, isaac_lab

    @staticmethod
    def _retrieve_mlflow_tracking_uri_from_aml() -> str:
        azure_subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID", os.getenv("AZUREML_RUN_SUBSCRIPTION_ID"))
        azure_resource_group = os.getenv("AZURE_RESOURCE_GROUP", os.getenv("AZUREML_RUN_RESOURCE_GROUP"))
        azure_workspace_name = os.getenv("AZURE_WORKSPACE_NAME", os.getenv("AZUREML_RUN_WORKSPACE_NAME"))

        from azure.ai.ml import MLClient
        from azure.identity import AzureCliCredential

        credential = AzureCliCredential()

        if all([azure_subscription_id, azure_resource_group, azure_workspace_name]):
            ml_client = MLClient(
                credential=credential,
                subscription_id=azure_subscription_id,
                resource_group_name=azure_resource_group,
                workspace_name=azure_workspace_name,
            )
        else:
            try:
                ml_client = MLClient.from_config(credential)
            except Exception as e:
                raise ValueError(
                    "Could not resolve Azure Machine Learning details. Either provide a config.json file or set environment variables."
                ) from e

        return ml_client.workspaces.get(azure_workspace_name).mlflow_tracking_uri

    @staticmethod
    def create(policy: PolicyInfo, task: str, *, params: dict[str, Any] | None = None) -> _EvaluationMetricsLogger:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not mlflow_tracking_uri:
            mlflow_tracking_uri = _EvaluationMetricsLogger._retrieve_mlflow_tracking_uri_from_aml()

        isaac_sim, isaac_lab = _EvaluationMetricsLogger._get_isaac_versions()
        params = params or {}
        params.update(
            {
                "isaac_sim": isaac_sim,
                "isaac_lab": isaac_lab,
            }
        )

        return _EvaluationMetricsLogger(
            policy=policy,
            task=task,
            mlflow_tracking_uri=mlflow_tracking_uri,
            params=params,
        )

class MLflowLoggerWrapper(gym.Wrapper):
    """Wrapper for logging metrics to MLflow.

    Args:
        gym (gym.Env): The environment to wrap.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        policy: PolicyInfo,
        task: str,
        task_description: str,
        number_of_episodes: int,
        artifact_paths: list[os.PathLike | None] | None = None,
        metrics_observation_group_name: str | None = None,
    ):
        super().__init__(env)

        unwrapped_env = env.unwrapped
        extra_params = {
            "env/num_envs": unwrapped_env.num_envs,
            "env/episode_length_s": unwrapped_env.cfg.episode_length_s,
            "env/decimation": unwrapped_env.cfg.decimation,
            "env/sim/render_interval": unwrapped_env.cfg.sim.render_interval,
            "env/sim/dt": unwrapped_env.cfg.sim.dt,
            "env/action_space": str(unwrapped_env.action_space),
            "env/seed": unwrapped_env.cfg.seed,
            "env/task": task,
            "env/step_dt": unwrapped_env.step_dt,
            "number_of_episodes": number_of_episodes,
            "task_description": task_description,
        }

        self.metrics_observation_group_name = metrics_observation_group_name or "metrics"
        self.metrics_logger = _EvaluationMetricsLogger.create(policy=policy, task=task, params=extra_params)
        self.metrics_logger.start()

        self.observation_for_metrics = env.unwrapped.observation_manager.active_terms.get(
            self.metrics_observation_group_name, []
        )
        self.last_obs: dict[str, Any] | None = None

        self.artifact_paths = artifact_paths or []
        self.step_dt = unwrapped_env.step_dt

    def step(self, action: torch.Tensor | Any) -> tuple[Any, float, bool, bool, dict]:
        # Need to make a copy to avoid resetting step_number on env.step
        self.step_number = self.env.unwrapped.episode_length_buf.clone()
        obs, reward, done, truncated, info = self.env.step(action)
        termination_status = detect_termination(done=done, truncated=truncated, extras=info)
        if termination_status.terminated:

            extra_metrics = self.extract_metrics(self.last_obs or obs)

            # Currently MLflowLogger supports only single environment logging, that is why
            # we log only the first environment's duration
            if not termination_status.success:
                self.metrics_logger.log_failed_episode(
                    ",".join(termination_status.reasons) if termination_status.reasons else "unknown",
                    (self.step_number[0].item()+1)*self.step_dt,
                    extra_metrics=extra_metrics,
                )
            else:
                self.metrics_logger.log_successful_episode(
                    (self.step_number[0].item()+1)*self.step_dt,
                    extra_metrics=extra_metrics,
                )

        # Capture last observation for metrics extraction on reset
        # During a reset, the observation returned corresponds to the new episode
        self.last_obs = obs

        return obs, reward, done, truncated, info

    def close(self):
        """Closes the wrapper and :attr:`env`."""
        super().close()

        if self.metrics_logger:
            if self.artifact_paths:
                self.metrics_logger.log_artifacts(self.artifact_paths)
            self.metrics_logger.close()
            self.metrics_logger = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_start = time.time()

        return obs, info

    def extract_metrics(self, obs: dict) -> dict[str, float]:
        metrics = {}
        for metric_name in self.observation_for_metrics:
            metric_value = obs[self.metrics_observation_group_name][metric_name]
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            metrics[metric_name] = float(metric_value)
        return metrics

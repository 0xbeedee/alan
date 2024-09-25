import argparse
import os
from typing import Dict, Any

import torch
import gymnasium as gym
import tianshou as ts
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from models import SelfModel, EnvModel
from intrinsic import ICM
from utils import EpochStatsPlotter
from config import ConfigManager
from utils.experiment import ExperimentFactory

DEFAULT_DEVICE = "cpu"
LOG_DIR = "logs"


def setup_config(
    base_config_path: str, env_config: str, policy_config: str, obsnet_config: str
) -> ConfigManager:
    """Sets up and validate the configuration."""
    config = ConfigManager(base_config_path)
    config.create_config(
        {
            "environment": env_config,
            "policy": policy_config,
            "obsnet": obsnet_config,
        }
    )
    # TODO a config check? notify the user if he made some strange choices along the way (like using a DiscreteObsNet with Dict spaces, e.g.,)
    return config


def setup_environment(config: ConfigManager) -> gym.Env:
    """Sets up the gym environment."""
    env_name = config.get("environment.name")
    try:
        env = gym.make(env_name)
        return env
    except gym.error.Error as e:
        raise RuntimeError(f"Failed to create environment {env_name}: {e}")


def setup_vector_envs(
    env: gym.Env, config: ConfigManager
) -> tuple[ts.env.DummyVectorEnv, ts.env.DummyVectorEnv]:
    """Sets up vector environments for training and testing."""
    num_train_envs = config.get("environment.num_train_envs")
    num_test_envs = config.get("environment.num_test_envs")
    train_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(num_test_envs)])
    return train_envs, test_envs


def setup_buffers(
    config: ConfigManager,
    num_train_envs: int,
    num_test_envs: int,
    factory: ExperimentFactory,
) -> tuple[Any, Any]:
    """Sets up replay buffers for training and testing."""
    train_buf_size = config.get("buffers.train_buf_size")
    test_buf_size = config.get("buffers.test_buf_size")
    train_buf = factory.create_buffer(train_buf_size, num_train_envs)
    test_buf = factory.create_buffer(test_buf_size, num_test_envs)
    return train_buf, test_buf


def setup_networks(
    factory: ExperimentFactory, env: gym.Env, device: torch.device
) -> tuple[Any, Any, Any]:
    """Sets up observation, actor, and critic networks."""
    obs_net = factory.create_obsnet()
    actor_net, critic_net = factory.create_actor_critic(
        obs_net, env.action_space, device
    )
    return obs_net, actor_net, critic_net


def setup_models(
    obs_net: Any, env: gym.Env, train_buf: Any, device: torch.device
) -> tuple[EnvModel, SelfModel]:
    """Sets up environment and self models."""
    # TODO can't I use YAML for more flexibility here? re-think this when I introduce other intrinsic modules
    env_model = EnvModel()
    self_model = SelfModel(
        obs_net, env.action_space, train_buf, ICM, her_horizon=3, device=device
    )
    return env_model, self_model


def setup_policy(
    factory: ExperimentFactory,
    self_model: SelfModel,
    env_model: EnvModel,
    actor_net: Any,
    critic_net: Any,
    optimizer: torch.optim.Optimizer,
    env: gym.Env,
    device: torch.device,
) -> Any:
    """Sets up the policy."""
    return factory.create_policy(
        self_model,
        env_model,
        actor_net,
        critic_net,
        optimizer,
        env.action_space,
        env.observation_space,
        False,  # no action scaling
        device,
    )


def setup_collectors(
    factory: ExperimentFactory,
    policy: Any,
    train_envs: ts.env.DummyVectorEnv,
    test_envs: ts.env.DummyVectorEnv,
    train_buf: Any,
    test_buf: Any,
) -> tuple[Any, Any]:
    """Sets up collectors for training and testing."""
    train_collector = factory.create_collector(policy, train_envs, train_buf)
    test_collector = factory.create_collector(policy, test_envs, test_buf)
    return train_collector, test_collector


def setup_logger(env_name: str, is_goal_aware: bool) -> TensorboardLogger:
    """Sets up the TensorboardLogger."""
    log_path = os.path.join(
        LOG_DIR, env_name.lower(), "goal" if is_goal_aware else "vanilla"
    )
    writer = SummaryWriter(log_path)
    return TensorboardLogger(writer)


def run_experiment(trainer: Any) -> list[Dict[str, Any]]:
    """Runs the experiment and collect epoch statistics."""
    epoch_stats = []
    for epoch_stat in trainer:
        epoch_stats.append(epoch_stat)
    return epoch_stats


def plot_results(factory: ExperimentFactory, epoch_stats: list[Dict[str, Any]]) -> None:
    """Plots the results if applicable."""
    # TODO should probably adjust the plotter to make it work with vanilla Tianshou policies as well
    if factory.is_goal_aware:
        plotter = EpochStatsPlotter(epoch_stats)
        plotter.plot(figsize=(8, 8))


def main(
    base_config_path: str,
    env_config: str,
    policy_config: str,
    obsnet_config: str,
    device: torch.device,
) -> None:
    config = setup_config(base_config_path, env_config, policy_config, obsnet_config)
    factory = ExperimentFactory(config)

    env = setup_environment(config)
    env = factory.wrap_env(env)

    train_envs, test_envs = setup_vector_envs(env, config)
    train_buf, test_buf = setup_buffers(
        config, len(train_envs), len(test_envs), factory
    )

    obs_net, actor_net, critic_net = setup_networks(factory, env, device)
    env_model, self_model = setup_models(obs_net, env, train_buf, device)

    lr = config.get("policy.learning_rate")
    combined_params = set(list(actor_net.parameters()) + list(critic_net.parameters()))
    optimizer = torch.optim.Adam(combined_params, lr=lr)

    policy = setup_policy(
        factory, self_model, env_model, actor_net, critic_net, optimizer, env, device
    )

    train_collector, test_collector = setup_collectors(
        factory, policy, train_envs, test_envs, train_buf, test_buf
    )

    logger = setup_logger(config.get("environment.name"), factory.is_goal_aware)

    trainer = factory.create_trainer(
        policy, train_collector, test_collector, logger, device
    )

    epoch_stats = run_experiment(trainer)
    plot_results(factory, epoch_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            The Reinforcement Learning Experiment Runner.

            This script sets up and runs reinforcement learning experiments with customizable environments, policies, and observation networks.

            Experiments are defined through YAML configuration files (usually located in the config/ directory), allowing for easy parameter tuning and reproducibility.
            """
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the base configuration file (including .yaml extension).",
        metavar="BASE_CONFIG_PATH",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        required=True,
        help="Name of the environment config file (without .yaml extension)",
        metavar="ENV_CONFIG",
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        required=True,
        help="Name of the policy config file (without .yaml extension)",
        metavar="POLICY_CONFIG",
    )
    parser.add_argument(
        "-o",
        "--obsnet",
        type=str,
        required=True,
        help="Name of the observation network config file (without .yaml extension)",
        metavar="OBSNET_CONFIG",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    main(args.config, args.env, args.policy, args.obsnet, device=device)

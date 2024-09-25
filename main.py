import argparse
import os

from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
import torch
import tianshou as ts
import gymnasium as gym

from models import SelfModel, EnvModel
from intrinsic import ICM
from utils import EpochStatsPlotter
from config import ConfigManager

from utils.experiment import *


def main(
    base_config_path: str,
    env_config: str,
    policy_config: str,
    obsnet_config: str,
    device: torch.device,
):
    config = ConfigManager(base_config_path)
    config.create_config(
        {
            "environment": env_config,
            "policy": policy_config,
            "obsnet": obsnet_config,
        }
    )

    is_goal_aware = config.get("is_goal_aware")
    env_name = config.get("environment.name")
    num_train_envs = config.get("environment.num_train_envs")
    num_test_envs = config.get("environment.num_test_envs")
    train_buf_size = config.get("buffers.train_buf_size")
    test_buf_size = config.get("buffers.test_buf_size")
    policy_name = config.get("policy.name")
    lr = config.get("policy.learning_rate")
    log_path = os.path.join(
        "logs", env_name.lower(), "goal" if is_goal_aware else "no_goal"
    )

    env = gym.make(env_name)
    env = wrap_env(env, is_goal_aware=is_goal_aware)

    train_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(num_test_envs)])

    train_buf = choose_buffer(
        train_buf_size, num_train_envs, is_goal_aware=is_goal_aware
    )
    test_buf = choose_buffer(test_buf_size, num_test_envs, is_goal_aware=is_goal_aware)

    obs_net = choose_obsnet(env, config)
    actor_net, critic_net = choose_actor_critic(
        obs_net, env.action_space, device, is_goal_aware=is_goal_aware
    )

    # TODO add options in config for intrinsic modules (selfmodel and envmodel are always the same, so no need)
    env_model = EnvModel()
    self_model = SelfModel(
        obs_net, env.action_space, train_buf, ICM, her_horizon=3, device=device
    )

    combined_params = set(list(actor_net.parameters()) + list(critic_net.parameters()))
    optimizer = torch.optim.Adam(combined_params, lr=lr)

    policy = choose_policy(
        policy_name,
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

    train_collector = choose_collector(
        policy, train_envs, train_buf, is_goal_aware=is_goal_aware
    )
    test_collector = choose_collector(
        policy, test_envs, test_buf, is_goal_aware=is_goal_aware
    )

    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    trainer = choose_trainer(
        policy,
        train_collector,
        test_collector,
        logger,
        device,
        config,
        is_goal_aware=is_goal_aware,
    )

    epoch_stats = []
    for epoch_stat in trainer:
        epoch_stats.append(epoch_stat)

    # TODO this plotter doesn't work with vanilla tianshou policies
    plotter = EpochStatsPlotter(epoch_stats)
    plotter.plot(figsize=(8, 8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The entrypoint for thorough experimentation."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Relateive path to the base configuration file (including the .yaml extension).",
        metavar="BASE_CONFIG_PATH",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        required=True,
        help="Name of the file containing the YAML config for the environment (without the .yaml extension)",
        metavar="ENV_CONFIG",
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        required=True,
        help="Name of the file containing the YAML config for the policy (without the .yaml extension)",
        metavar="POLICY_CONFIG",
    )
    parser.add_argument(
        "-o",
        "--obsnet",
        type=str,
        required=True,
        help="Name of the file containing the YAML config for the observation network (without the .yaml extension)",
        metavar="OBSNET_CONFIG",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    main(
        args.config,
        args.env,
        args.policy,
        args.obsnet,
        device=device,
    )

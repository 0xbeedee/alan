import argparse

from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

import torch
import tianshou as ts

import gymnasium as gym

from networks import DiscreteObsNet
from models import SelfModel, EnvModel
from intrinsic import ICM
from utils import EpochStatsPlotter
from utils.experiment import *
from config import Config


def main(config_file, device):
    config = Config(config_file)
    is_goal_aware = config.get("is_goal_aware")

    env = gym.make(config.get("environment.name"))
    env = wrap_env(env, is_goal_aware=is_goal_aware)

    num_train_envs = config.get("environment.num_train_envs")
    num_test_envs = config.get("environment.num_test_envs")
    train_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(num_test_envs)])

    train_buf_size = config.get("buffers.train_buf_size")
    test_buf_size = config.get("buffers.test_buf_size")
    train_buf = choose_buffer(
        train_buf_size, num_train_envs, is_goal_aware=is_goal_aware
    )
    test_buf = choose_buffer(test_buf_size, num_test_envs, is_goal_aware=is_goal_aware)

    # obs_net should probably be treated somewhat like actor and critic (I also don't always need an actor and a critic, btw)
    obs_net = DiscreteObsNet([16], 10)
    actor_net, critic_net = choose_actor_critic(
        obs_net, env.action_space, device, is_goal_aware=is_goal_aware
    )

    # TODO add options in config for intrinsic modules (selfmodel and envmodel are always the same, so no need)
    env_model = EnvModel()
    self_model = SelfModel(
        obs_net, env.action_space, train_buf, ICM, her_horizon=3, device=device
    )

    combined_params = set(list(actor_net.parameters()) + list(critic_net.parameters()))
    optimizer = torch.optim.Adam(
        combined_params, lr=config.get("algorithm.learning_rate")
    )

    algorithm_name = config.get("algorithm.name")
    policy = choose_policy(
        algorithm_name,
        self_model,
        env_model,
        actor_net,
        critic_net,
        optimizer,
        env.action_space,
        env.observation_space,
        False,
        device,
    )

    train_collector = choose_collector(
        policy, train_envs, train_buf, is_goal_aware=is_goal_aware
    )
    test_collector = choose_collector(
        policy, test_envs, test_buf, is_goal_aware=is_goal_aware
    )

    writer = SummaryWriter(config.get("logging.log_path"))
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
    # TODO configure parser
    parser = argparse.ArgumentParser(
        description="The entrypoint for thourough experimentation."
    )
    parser.add_argument("config_file", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # TODO more flexibility in managing devices => more flexibility in CLI handling in general
    # (should probably pass all the args to the main() function directly)
    main(args.config_file, device=torch.device("cpu"))

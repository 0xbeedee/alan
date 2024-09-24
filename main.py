import argparse

from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

import torch
import tianshou as ts

import os
from datetime import datetime
import gymnasium as gym

from environments import DictObservation
from networks import DiscreteObsNet, GoalNetHackActor, GoalNetHackCritic
from policies import PPOBasedPolicy
from models import SelfModel, EnvModel
from intrinsic import ICM
from core import GoalCollector, GoalVectorReplayBuffer, GoalOnpolicyTrainer
from utils import EpochStatsPlotter
from config import Config


def main(config_file, device):
    config = Config(config_file)

    env = gym.make(config.get("environment.name"))
    # TODO more flexibility?
    env = DictObservation(env)

    num_train_envs = config.get("environment.num_train_envs")
    num_test_envs = config.get("environment.num_test_envs")
    train_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(num_test_envs)])

    train_buf_size = config.get("buffers.train_buf_size")
    test_buf_size = config.get("buffers.test_buf_size")
    train_buf = GoalVectorReplayBuffer(train_buf_size, num_train_envs)
    test_buf = GoalVectorReplayBuffer(test_buf_size, num_train_envs)

    # TODO more flexibility?
    obs_net = DiscreteObsNet([16], 10)
    actor_net = GoalNetHackActor(obs_net, env.action_space, device=device)
    critic_net = GoalNetHackCritic(obs_net, device=device)

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
    if algorithm_name == "PPOBased":
        policy = PPOBasedPolicy(
            self_model=self_model,
            env_model=env_model,
            act_net=actor_net,
            critic_net=critic_net,
            optim=optimizer,
            action_space=env.action_space,
            observation_space=env.observation_space,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    train_collector = GoalCollector(policy, train_envs, train_buf)
    test_collector = GoalCollector(policy, test_envs, test_buf)

    # TODO more flexibility here? possibly specify log in yaml?
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    log_path = os.path.join("../logs", "simplenv", timestamp)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # TODO more flexibility
    trainer = GoalOnpolicyTrainer(
        **config.get("training"),
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        repeat_per_collect=1,
        logger=logger,
        device=device,
    )

    epoch_stats = []
    for epoch_stat in trainer:
        epoch_stats.append(epoch_stat)

    plotter = EpochStatsPlotter(epoch_stats)
    plotter.plot(figsize=(8, 8))


if __name__ == "__main__":
    # TODO configure parser
    parser = argparse.ArgumentParser(description="RL Testing Harness")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # TODO more flexibility in managing devices => more flexibility in CLI handling in general
    # (should probably pass all the args to the main() function directly)
    main(args.config, device=torch.device("cpu"))

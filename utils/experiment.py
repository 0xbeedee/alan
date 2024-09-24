from typing import Callable, Tuple, Union, Type
import gymnasium as gym
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer, OffpolicyTrainer, OfflineTrainer
from tianshou.env.venvs import BaseVectorEnv
from tianshou.utils import TensorboardLogger

import torch
from torch import nn

from environments import DictObservation, Resetting
from networks import *
from policies import PPOBasedPolicy
from intrinsic import ICM
from config import Config
from core import (
    GoalCollector,
    GoalVectorReplayBuffer,
    GoalOnpolicyTrainer,
    GoalOffpolicyTrainer,
    GoalOfflineTrainer,
    CorePolicy,
)
from core.types import SelfModelProtocol, EnvModelProtocol


def wrap_env(env: gym.Env, is_goal_aware: bool) -> gym.Env:
    observation_keys = (
        "glyphs",
        "chars",
        "colors",
        "specials",
        "blstats",
        "message",
        "inv_glyphs",
        "inv_strs",
        "inv_letters",
        "inv_oclasses",
        "screen_descriptions",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
    )

    # all the options below apply to goal aware environments
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return DictObservation(env)

    if all(obs_key in env.observation_space.keys() for obs_key in observation_keys):
        return Resetting(env)


def choose_buffer(
    buf_size: int, env_num: int, is_goal_aware: bool
) -> Union[GoalVectorReplayBuffer, VectorReplayBuffer]:
    buf_class = GoalVectorReplayBuffer if is_goal_aware else VectorReplayBuffer
    return buf_class(buf_size, env_num)


def choose_actor_critic(
    obs_net: nn.Module,
    action_space: gym.Space,
    device: torch.device,
    is_goal_aware: bool,
) -> Tuple[
    Union[GoalNetHackActor, SimpleNetHackActor],
    Union[GoalNetHackCritic, SimpleNetHackCritic],
]:
    actor_class, critic_class = (
        (GoalNetHackActor, GoalNetHackCritic)
        if is_goal_aware
        else (SimpleNetHackActor, SimpleNetHackCritic)
    )
    return actor_class(obs_net, action_space, device=device), critic_class(
        obs_net, device=device
    )


def choose_policy(
    algorithm_name: str,
    self_model: SelfModelProtocol,
    env_model: EnvModelProtocol,
    act_net: Union[GoalNetHackActor, SimpleNetHackActor],
    critic_net: Union[GoalNetHackCritic, SimpleNetHackCritic],
    optim: torch.optim.Optimizer,
    action_space: gym.Space,
    observation_space: gym.Space,
    action_scaling: bool,
    device: torch.device,
    dist_fn: Callable = None,
) -> Union[CorePolicy, BasePolicy]:

    # TODO should this be here?
    def dist_fn(logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)

    policy_map = {
        "ppo_based": lambda: PPOBasedPolicy(
            self_model=self_model,
            env_model=env_model,
            act_net=act_net,
            critic_net=critic_net,
            optim=optim,
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            device=device,
        ),
        "ppo": lambda: PPOPolicy(
            actor=act_net,
            critic=critic_net,
            optim=optim,
            action_space=action_space,
            action_scaling=action_scaling,
            dist_fn=dist_fn,
        ),
    }

    if algorithm_name not in policy_map:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    return policy_map[algorithm_name]()


def choose_collector(
    policy: Union[CorePolicy, BasePolicy],
    envs: BaseVectorEnv,
    buffer: Union[GoalVectorReplayBuffer, VectorReplayBuffer],
    is_goal_aware: bool,
) -> Union[Collector, GoalCollector]:
    collector_class = GoalCollector if is_goal_aware else Collector
    return collector_class(policy, envs, buffer)


def choose_trainer(
    policy: BasePolicy,
    train_collector: Union[Collector, GoalCollector],
    test_collector: Union[Collector, GoalCollector],
    logger: TensorboardLogger,
    device: torch.device,
    config: Config,
    is_goal_aware: bool,
    trainer_type: str = "onpolicy",
) -> Union[
    OnpolicyTrainer,
    OffpolicyTrainer,
    OfflineTrainer,
    GoalOnpolicyTrainer,
    GoalOffpolicyTrainer,
    GoalOfflineTrainer,
]:
    trainer_map = {
        "onpolicy": (OnpolicyTrainer, GoalOnpolicyTrainer),
        "offpolicy": (OffpolicyTrainer, GoalOffpolicyTrainer),
        "offline": (OfflineTrainer, GoalOfflineTrainer),
    }

    if trainer_type not in trainer_map:
        raise ValueError(
            f"Invalid trainer_type: {trainer_type}. Must be one of {list(trainer_map.keys())}"
        )

    standard_trainer, goal_trainer = trainer_map[trainer_type]
    trainer_class = goal_trainer if is_goal_aware else standard_trainer

    common_kwargs = {
        "policy": policy,
        "logger": logger,
        **config.get("training", {}),
    }
    if is_goal_aware:
        common_kwargs["device"] = device
    if trainer_type != "offline":
        common_kwargs.update(
            {
                "train_collector": train_collector,
                "test_collector": test_collector,
            }
        )

    return trainer_class(**common_kwargs)

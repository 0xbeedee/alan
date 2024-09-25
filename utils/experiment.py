import gymnasium as gym
import torch
from torch import nn
from typing import Tuple, Union
from tianshou.data import VectorReplayBuffer, Collector, EpochStats
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer, OffpolicyTrainer, OfflineTrainer
from tianshou.env.venvs import BaseVectorEnv
from tianshou.utils import TensorboardLogger

from environments import DictObservation, Resetting
from networks import (
    NetHackObsNet,
    DiscreteObsNet,
    GoalNetHackActor,
    SimpleNetHackActor,
    GoalNetHackCritic,
    SimpleNetHackCritic,
)
from policies import PPOBasedPolicy
from config import ConfigManager
from core import (
    GoalCollector,
    GoalVectorReplayBuffer,
    GoalOnpolicyTrainer,
    GoalOffpolicyTrainer,
    GoalOfflineTrainer,
    CorePolicy,
)
from core.types import SelfModelProtocol, EnvModelProtocol
from utils.plotters import GoalStatsPlotter, VanillaStatsPlotter


class ExperimentFactory:
    """A class used for providing a simple, intuitive and flexible interface for creating experiments."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.is_goal_aware = config.get("is_goal_aware")

    def wrap_env(self, env: gym.Env) -> gym.Env:
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

        if isinstance(env.observation_space, gym.spaces.Discrete):
            return DictObservation(env)

        if all(obs_key in env.observation_space.keys() for obs_key in observation_keys):
            return Resetting(env)

        return env

    def create_buffer(
        self, buf_size: int, env_num: int
    ) -> Union[GoalVectorReplayBuffer, VectorReplayBuffer]:
        buf_class = GoalVectorReplayBuffer if self.is_goal_aware else VectorReplayBuffer
        return buf_class(buf_size, env_num)

    def create_obsnet(self) -> nn.Module:
        obs_net_map = {"nethack": NetHackObsNet, "discrete": DiscreteObsNet}
        obs_class = obs_net_map[self.config.get("obsnet.name")]
        return obs_class(**self.config.get_except("obsnet", exclude="name"))

    def create_actor_critic(
        self, obs_net: nn.Module, action_space: gym.Space, device: torch.device
    ) -> Tuple[
        Union[GoalNetHackActor, SimpleNetHackActor],
        Union[GoalNetHackCritic, SimpleNetHackCritic],
    ]:
        actor_class, critic_class = (
            (GoalNetHackActor, GoalNetHackCritic)
            if self.is_goal_aware
            else (SimpleNetHackActor, SimpleNetHackCritic)
        )
        return actor_class(obs_net, action_space, device=device), critic_class(
            obs_net, device=device
        )

    def create_policy(
        self,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        act_net: Union[GoalNetHackActor, SimpleNetHackActor],
        critic_net: Union[GoalNetHackCritic, SimpleNetHackCritic],
        optim: torch.optim.Optimizer,
        action_space: gym.Space,
        observation_space: gym.Space,
        action_scaling: bool,
        device: torch.device,
    ) -> Union[CorePolicy, BasePolicy]:

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
                dist_fn=_ppo_discrete_dist_fn,
            ),
        }

        policy_name = self.config.get("policy.name")
        if policy_name not in policy_map:
            raise ValueError(f"Unsupported algorithm: {policy_name}")

        return policy_map[policy_name]()

    def create_collector(
        self,
        policy: Union[CorePolicy, BasePolicy],
        envs: BaseVectorEnv,
        buffer: Union[GoalVectorReplayBuffer, VectorReplayBuffer],
    ) -> Union[Collector, GoalCollector]:
        collector_class = GoalCollector if self.is_goal_aware else Collector
        return collector_class(policy, envs, buffer)

    def create_trainer(
        self,
        policy: BasePolicy,
        train_collector: Union[Collector, GoalCollector],
        test_collector: Union[Collector, GoalCollector],
        logger: TensorboardLogger,
        device: torch.device,
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
        trainer_class = goal_trainer if self.is_goal_aware else standard_trainer

        common_kwargs = {
            "policy": policy,
            "logger": logger,
            **self.config.get("training", {}),
        }
        if self.is_goal_aware:
            common_kwargs["device"] = device
        if trainer_type != "offline":
            common_kwargs.update(
                {
                    "train_collector": train_collector,
                    "test_collector": test_collector,
                }
            )

        return trainer_class(**common_kwargs)

    def create_plotter(
        self, epoch_stats: EpochStats
    ) -> GoalStatsPlotter | VanillaStatsPlotter:
        plt_class = GoalStatsPlotter if self.is_goal_aware else VanillaStatsPlotter
        return plt_class(epoch_stats)


def _ppo_discrete_dist_fn(logits: torch.Tensor):
    return torch.distributions.Categorical(logits=logits)

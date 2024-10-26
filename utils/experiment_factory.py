from typing import Tuple, Union

import gymnasium as gym
import torch
from torch import nn
from tianshou.data import VectorReplayBuffer, Collector, EpochStats
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer, OffpolicyTrainer, OfflineTrainer
from tianshou.env.venvs import BaseVectorEnv
from tianshou.utils import TensorboardLogger

from environments import DictObservation, Resetting, RecordTTY, RecordRGB
from networks import (
    ObsNet,
    GoalNetHackActor,
    SimpleNetHackActor,
    GoalNetHackCritic,
    SimpleNetHackCritic,
    NetHackVAE,
    MDNRNN,
)
from intrinsic import ICM, ZeroICM, DeltaICM, HER, ZeroHER
from models import NetHackVAETrainer, MDNRNNTrainer
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

    def wrap_env(self, env: gym.Env, rec_path: str) -> gym.Env:
        wrapped_env = None
        if isinstance(env.observation_space, gym.spaces.Discrete):
            wrapped_env = DictObservation(env)
        elif "NetHack" in env.unwrapped.spec.id:
            wrapped_env = Resetting(env)

        return (
            RecordRGB(wrapped_env, output_path=rec_path)
            if env.render_mode == "rgb_array"
            else RecordTTY(wrapped_env, output_path=rec_path)
        )

    def create_buffer(
        self, buf_size: int, env_num: int
    ) -> Union[GoalVectorReplayBuffer, VectorReplayBuffer]:
        buf_class = GoalVectorReplayBuffer if self.is_goal_aware else VectorReplayBuffer
        return buf_class(buf_size, env_num)

    def create_vae_mdnrnn(self, observation_space: gym.Space, device: torch.device):
        # TODO add DiscreteVAE and MDNRNN (? => the MDNRNN could probably always be the same, as long as the shapes match)
        vae = NetHackVAE(
            **self.config.get("obsnet.vae"),
            observation_space=observation_space,
            device=device,
        )
        mdnrnn = MDNRNN(
            **self.config.get("obsnet.mdnrnn"),
            device=device,
        )
        return vae, mdnrnn

    def create_obsnet(self, vae_encoder: nn.Module, device: torch.device) -> nn.Module:
        # the vae_encoder is the part that adapts to the environment
        return ObsNet(vae_encoder=vae_encoder, device=device)

    def create_intrinsic_modules(
        self,
        obs_net: nn.Module,
        action_space: gym.Space,
        buf: GoalVectorReplayBuffer,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
    ):
        fast_intrinsic_map = {"icm": ICM, "zero_icm": ZeroICM, "delta_icm": DeltaICM}
        slow_intrinsic_map = {"her": HER, "zero_her": ZeroHER}

        fast_intrinsic_class = fast_intrinsic_map[
            self.config.get("intrinsic.fast.name")
        ]
        slow_intrinsic_class = slow_intrinsic_map[
            self.config.get("intrinsic.slow.name")
        ]

        fast_intrinsic_module = fast_intrinsic_class(
            obs_net,
            action_space,
            batch_size,
            learning_rate,
            **self.config.get_except("intrinsic.fast", exclude="name"),
            device=device,
        )
        slow_intrinsic_module = slow_intrinsic_class(
            buf,
            **self.config.get_except("intrinsic.slow", exclude="name"),
        )

        return fast_intrinsic_module, slow_intrinsic_module

    def create_envmodel_trainers(
        self,
        vae: nn.Module,
        mdnrnn: nn.Module,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
    ) -> Tuple[nn.Module, nn.Module]:
        vae_trainer = NetHackVAETrainer(
            vae, batch_size, learning_rate=learning_rate, device=device
        )
        mdnrnn_trainer = MDNRNNTrainer(
            mdnrnn,
            vae,
            batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        return vae_trainer, mdnrnn_trainer

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
            ),
            "ppo": lambda: PPOPolicy(
                actor=act_net,
                critic=critic_net,
                optim=optim,
                action_space=action_space,
                action_scaling=action_scaling,
                # continuous action spaces are out of scope for our work
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
        trainer_type: str = "onpolicy",
        is_dream: bool = False,
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

        train_type = "dream" if is_dream else "real"
        common_kwargs = {
            "policy": policy,
            "logger": logger,
            **self.config.get(f"training.{train_type}", {}),
        }

        if trainer_type != "offline":
            common_kwargs.update(
                {
                    "train_collector": train_collector,
                    "test_collector": test_collector,
                }
            )

        return trainer_class(**common_kwargs)

    def create_plotter(
        self,
        epoch_stats: EpochStats,
    ) -> GoalStatsPlotter | VanillaStatsPlotter:
        plt_class = GoalStatsPlotter if self.is_goal_aware else VanillaStatsPlotter
        return plt_class(epoch_stats)


def _ppo_discrete_dist_fn(logits: torch.Tensor):
    return torch.distributions.Categorical(logits=logits)

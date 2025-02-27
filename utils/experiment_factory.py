from typing import Tuple, Union
from core.types import SelfModelProtocol, EnvModelProtocol

import gymnasium as gym
import torch
from torch import nn
from tianshou.data import VectorReplayBuffer, EpochStats
from tianshou.policy import BasePolicy
from tianshou.env.venvs import BaseVectorEnv
from tianshou.utils import TensorboardLogger

from datetime import datetime
import os

from environments import DictObservation, Resetting, RecordTTY, RecordRGB
from networks import (
    ObsNet,
    GoalNetHackActor,
    GoalNetHackCritic,
    NetHackVAE,
    DiscreteVAE,
    MDNRNN,
)
from intrinsic import ICM, ZeroICM, DeltaICM, BBold, ZeroBBold, HER, ZeroHER
from models.trainers import NetHackVAETrainer, MDNRNNTrainer, DiscreteVAETrainer
from policies import GoalPPO, RandomPolicy
from config import ConfigManager
from core import (
    GoalCollector,
    GoalVectorReplayBuffer,
    GoalOnpolicyTrainer,
    GoalOffpolicyTrainer,
    GoalOfflineTrainer,
    CorePolicy,
)
from lifelong import VectorKnowledgeBase, TrajectoryBandit
from .plotter import Plotter


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

    def create_buffer(self, buf_size: int, env_num: int) -> GoalVectorReplayBuffer:
        return GoalVectorReplayBuffer(buf_size, env_num)

    def create_knowledge_base_and_bandit(
        self, kb_size: int, env_num: int, kb_path: str
    ) -> Tuple[VectorKnowledgeBase, TrajectoryBandit]:
        if os.path.exists(kb_path):
            # if the path exists, there must be at least one file in it
            dt = "%d%m%Y-%H%M%S"
            valid_kbs = [
                (f, datetime.strptime(f[:-3], dt)) for f in os.listdir(kb_path)
            ]
            latest_kb = max(valid_kbs, key=lambda x: x[1])[0]
            knowledge_base = VectorKnowledgeBase.load_hdf5(
                os.path.join(kb_path, latest_kb)
            )
        else:
            knowledge_base = VectorKnowledgeBase(kb_size, env_num)
        bandit = TrajectoryBandit(knowledge_base)
        return knowledge_base, bandit

    def create_vae_mdnrnn(self, observation_space: gym.Space, device: torch.device):
        vae_map = {
            "nethack": NetHackVAE,
            "discrete": DiscreteVAE,
        }

        vae_class = vae_map[self.config.get("obsnet.name")]
        vae = vae_class(
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
        # ObsNet is just a wrapper for the VAE encoder
        return ObsNet(vae_encoder=vae_encoder, device=device)

    def create_actor_critic(
        self,
        obs_net: nn.Module,
        state_dim: int,
        action_space: gym.Space,
        device: torch.device,
    ) -> Tuple[GoalNetHackActor, GoalNetHackCritic]:
        return GoalNetHackActor(
            obs_net, state_dim, action_space, device=device
        ), GoalNetHackCritic(obs_net, device=device)

    def create_intrinsic_modules(
        self,
        obs_net: nn.Module,
        action_space: gym.Space,
        buf: GoalVectorReplayBuffer,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
    ):
        fast_intrinsic_map = {
            "icm": ICM,
            "zero_icm": ZeroICM,
            "delta_icm": DeltaICM,
            "bebold": BBold,
            "zero_bebold": ZeroBBold,
        }
        slow_intrinsic_map = {"her": HER, "zero_her": ZeroHER}

        fast_intrinsic_class = fast_intrinsic_map[
            self.config.get("intrinsic_fast.name")
        ]
        slow_intrinsic_class = slow_intrinsic_map[
            self.config.get("intrinsic_slow.name")
        ]

        fast_intrinsic_module = fast_intrinsic_class(
            obs_net.o_dim,
            action_space.n,
            batch_size,
            learning_rate,
            **self.config.get_except("intrinsic_fast", exclude="name"),
            device=device,
        )
        slow_intrinsic_module = slow_intrinsic_class(
            obs_net,
            buf,
            **self.config.get_except("intrinsic_slow", exclude="name"),
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
        vae_trainer_map = {
            "nethack": NetHackVAETrainer,
            "discrete": DiscreteVAETrainer,
        }

        vae_trainer_class = vae_trainer_map[self.config.get("obsnet.name")]
        vae_trainer = vae_trainer_class(
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

    def create_policy(
        self,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        obs_net: nn.Module,
        act_net: GoalNetHackActor,
        critic_net: GoalNetHackCritic,
        optim: torch.optim.Optimizer,
        action_space: gym.Space,
        observation_space: gym.Space,
        action_scaling: bool,
    ) -> CorePolicy:

        policy_map = {
            "goal_ppo": lambda: GoalPPO(
                self_model=self_model,
                env_model=env_model,
                obs_net=obs_net,
                act_net=act_net,
                critic_net=critic_net,
                optim=optim,
                action_space=action_space,
                observation_space=observation_space,
                action_scaling=action_scaling,
            ),
            "random": lambda: RandomPolicy(
                self_model=self_model,
                env_model=env_model,
                obs_net=obs_net,
                action_space=action_space,
                observation_space=observation_space,
                action_scaling=action_scaling,
            ),
        }

        policy_name = self.config.get("policy.name")
        if policy_name not in policy_map:
            raise ValueError(f"Unsupported algorithm: {policy_name}")

        return policy_map[policy_name]()

    def create_collector(
        self,
        policy: CorePolicy,
        envs: BaseVectorEnv,
        buffer: GoalVectorReplayBuffer,
        knowledge_base: VectorReplayBuffer | None,
        bandit: TrajectoryBandit | None,
    ) -> GoalCollector:
        return GoalCollector(policy, envs, buffer, knowledge_base, bandit)

    def create_trainer(
        self,
        policy: BasePolicy,
        train_collector: GoalCollector,
        test_collector: GoalCollector,
        logger: TensorboardLogger,
        trainer_type: str = "onpolicy",
        is_dream: bool = False,
    ) -> Union[
        GoalOnpolicyTrainer,
        GoalOffpolicyTrainer,
        GoalOfflineTrainer,
    ]:
        trainer_map = {
            "onpolicy": GoalOnpolicyTrainer,
            "offpolicy": GoalOffpolicyTrainer,
            "offline": GoalOfflineTrainer,
        }

        if trainer_type not in trainer_map:
            raise ValueError(
                f"Invalid trainer_type: {trainer_type}. Must be one of {list(trainer_map.keys())}"
            )

        trainer_class = trainer_map[trainer_type]

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
    ) -> Plotter:
        return Plotter(epoch_stats)

from typing import Tuple, Union
from core.types import (
    SelfModelProtocol,
    EnvModelProtocol,
    FastIntrinsicModuleProtocol,
    SlowIntrinsicModuleProtocol,
)

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
    GoalActor,
    GoalCritic,
    NetHackVAE,
    DiscreteVAE,
    MDNRNN,
)
from intrinsic import ICM, ZeroICM, DeltaICM, BBold, ZeroBBold, HER, ZeroHER
from models.trainers import NetHackVAETrainer, MDNRNNTrainer, DiscreteVAETrainer
from models import SelfModel
from policies import GoalPPO, GoalDQN, RandomPolicy
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

    def wrap_env(self, env: gym.Env, rec_path: str | None = None) -> gym.Env:
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

    def create_vae_mdnrnn(
        self,
        observation_space: gym.Space,
        device: torch.device,
        weights_path: str | None = None,
    ) -> Tuple[Union[NetHackVAE, DiscreteVAE], MDNRNN, bool]:
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

        is_pretrained_vae = False
        is_pretrained_mdnrnn = False
        if weights_path and os.path.exists(weights_path):
            # load the weights from the specified path
            dt = "%d%m%Y-%H%M%S"
            valid_vae_weights = []
            valid_mdnrnn_weights = []
            for f in os.listdir(weights_path):
                try:
                    if f.endswith("_vae.pth"):
                        valid_vae_weights.append(
                            (f, datetime.strptime(f.split("_")[0], dt))
                        )
                    elif f.endswith("_mdnrnn.pth"):
                        valid_mdnrnn_weights.append(
                            (f, datetime.strptime(f.split("_")[0], dt))
                        )
                except ValueError:
                    continue

            if valid_vae_weights:
                is_pretrained_vae = True
                latest_vae_weights = max(valid_vae_weights, key=lambda x: x[1])[0]
                vae.load_state_dict(
                    torch.load(
                        os.path.join(weights_path, latest_vae_weights),
                        weights_only=True,
                    )
                )
            if valid_mdnrnn_weights:
                is_pretrained_mdnrnn = True
                latest_mdnrnn_weights = max(valid_mdnrnn_weights, key=lambda x: x[1])[0]
                mdnrnn.load_state_dict(
                    torch.load(
                        os.path.join(weights_path, latest_mdnrnn_weights),
                        weights_only=True,
                    )
                )

        return vae, mdnrnn, is_pretrained_vae and is_pretrained_mdnrnn

    def create_obsnet(self, vae_encoder: nn.Module, device: torch.device) -> nn.Module:
        # ObsNet is just a wrapper for the VAE encoder
        return ObsNet(vae_encoder=vae_encoder, device=device)

    def create_policy_nets(
        self,
        obs_net: nn.Module,
        state_dim: int,
        action_space: gym.Space,
        device: torch.device,
    ) -> Tuple[GoalActor, GoalCritic]:
        policy_config = self.config.get("policy")
        # all policies need an actor...
        actor = GoalActor(obs_net, state_dim, action_space, device)
        # ...but not all policies need a critic
        critic = (
            GoalCritic(obs_net, device) if policy_config["is_actor_critic"] else None
        )
        return actor, critic

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

    def create_self_model(
        self,
        fast_intrinsic_module: FastIntrinsicModuleProtocol,
        slow_intrinsic_module: SlowIntrinsicModuleProtocol,
    ) -> SelfModelProtocol:
        goal_strategy = self.config.get("selfmodel.goal_strategy", "random")
        goal_config = self.config.get_except("selfmodel", exclude="goal_strategy")

        self_model = SelfModel(
            fast_intrinsic_module,
            slow_intrinsic_module,
            goal_strategy=goal_strategy,
            **goal_config,
        )
        return self_model

    def create_envmodel_trainers(
        self,
        vae: nn.Module,
        mdnrnn: nn.Module,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
        use_finetuning: bool = False,
        freeze_envmodel: bool = False,
    ) -> Tuple[Union[NetHackVAETrainer, DiscreteVAETrainer], MDNRNNTrainer]:
        vae_trainer_map = {
            "nethack": NetHackVAETrainer,
            "discrete": DiscreteVAETrainer,
        }

        vae_trainer_class = vae_trainer_map[self.config.get("obsnet.name")]
        vae_trainer = vae_trainer_class(
            vae,
            batch_size,
            learning_rate=learning_rate,
            device=device,
            use_finetuning=use_finetuning,
            freeze_envmodel=freeze_envmodel,
            lr_scale=self.config.get("model_envmodel.lr_scale"),
        )

        mdnrnn_trainer = MDNRNNTrainer(
            mdnrnn,
            vae,
            batch_size,
            learning_rate=learning_rate,
            device=device,
            use_finetuning=use_finetuning,
            freeze_envmodel=freeze_envmodel,
            lr_scale=self.config.get("model_envmodel.lr_scale"),
        )

        return vae_trainer, mdnrnn_trainer

    def create_policy(
        self,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        obs_net: nn.Module,
        actor: GoalActor,
        critic: GoalCritic,
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
                act_net=actor,
                critic_net=critic,
                optim=optim,
                action_space=action_space,
                observation_space=observation_space,
                action_scaling=action_scaling,
            ),
            "goal_dqn": lambda: GoalDQN(
                self_model=self_model,
                env_model=env_model,
                obs_net=obs_net,
                model=actor,
                optim=optim,
                action_space=action_space,
                observation_space=observation_space,
                action_scaling=action_scaling,
                target_update_freq=self.config.get("policy.target_update_freq"),
                is_double=self.config.get("policy.is_double"),
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
        trainer_type: str,
        policy: BasePolicy,
        train_collector: GoalCollector,
        test_collector: GoalCollector,
        logger: TensorboardLogger,
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
            "train_collector": train_collector,
            "test_collector": test_collector,
            **self.config.get(f"training.{train_type}", {}),
        }
        return trainer_class(**common_kwargs)

    def create_plotter(
        self,
        epoch_stats: EpochStats,
    ) -> Plotter:
        return Plotter(epoch_stats)

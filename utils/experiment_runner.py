from typing import Generator
from contextlib import contextmanager
from datetime import datetime
import os

import torch
import gymnasium as gym
import tianshou as ts
from tianshou.data import EpochStats
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models import SelfModel, EnvModel
from intrinsic import ZeroICM, ZeroHER
from config import ConfigManager
from .experiment_factory import ExperimentFactory

from environments import DreamEnv

import warnings

# the version of pygame needed by Gymnasium still uses pkgdata, so we supress the deprecation warnings to avoid clutter
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"pygame\.pkgdata"
)

ART_DIR = "artefacts"
LOG_DIR = f"{ART_DIR}/logs"
PLOT_DIR = f"{ART_DIR}/plots"
REC_DIR = f"{ART_DIR}/recs"


class ExperimentRunner:
    def __init__(
        self,
        base_config_path: str,
        env_config: str,
        policy_config: str,
        obsnet_config: str,
        intrinsic_config: str,
        model_config: str,
        use_kb: bool,
        device: torch.device,
    ) -> None:
        self.base_config_path = base_config_path
        self.env_config = env_config
        self.policy_config = policy_config
        self.obsnet_config = obsnet_config
        self.intrinsic_config = intrinsic_config
        self.model_config = model_config
        self.use_kb = use_kb
        self.device = device

        self._setup_config()
        self.env_name = self.config.get("environment.base.name")
        self.num_envs = self.config.get("environment.vec.num_envs")
        self.num_dream_envs = self.config.get("environment.vec.num_dream_envs")

        self.train_buf_size = self.config.get("buffers.train_buf_size")
        self.dream_train_buf_size = self.config.get("buffers.dream_train_buf_size")
        self.test_buf_size = self.config.get("buffers.test_buf_size")
        self.kb_size = self.config.get("buffers.kb_size")

        # use the real batch size by default
        self.batch_size = self.config.get("training.real.batch_size")
        self.learning_rate = self.config.get("policy.learning_rate")

        self.factory = ExperimentFactory(self.config)
        self.is_goal_aware = self.factory.is_goal_aware

    def setup(self) -> None:
        print("[+] Setting up the environments...")
        self._setup_vector_envs(is_dream=False)
        print("[+] Setting up the buffers...")
        self._setup_buffers(is_dream=False)
        print("[+] Setting up the networks...")
        self._setup_networks()
        print("[+] Setting up the models...")
        self._setup_models()
        print("[+] Setting up the policy...")
        self._setup_policy()
        print("[+] Setting up the collectors...")
        self._setup_collectors(is_dream=False)
        print("[+] Setting up the logger...")
        self._setup_logger()
        print("[+] Setting up the trainer...")
        self._setup_trainer(is_dream=False)

        print("[+] Weaving the dream...")
        self._setup_dream()

    def run(self, save_pdf_plot: bool = True) -> None:
        """Runs the experiment and collects epoch statistics."""
        print("\n[+] Running the experiment...")
        self.epoch_stats, self.dream_epoch_stats = [], []
        for epoch_stat in self.trainer:
            self.epoch_stats.append(epoch_stat)
            if self._envmodel_is_good(epoch_stat):
                # only run the dream if we have a good enough model of the environment
                self._run_dream()

        print("\n[+] Plotting..." if not save_pdf_plot else "\n[+] Saving the plot...")
        self._plot(save_pdf=save_pdf_plot)

        print("[+] Recording a rollout...")
        self._record_rollout()

        print("[+] All done!")

    def _setup_config(self) -> None:
        """Sets up the configuration."""
        self.config = ConfigManager(self.base_config_path)
        self.config.create_config(
            {
                "environment": self.env_config,
                "policy": self.policy_config,
                "obsnet": self.obsnet_config,
                "intrinsic": self.intrinsic_config,
                "model": self.model_config,
            }
        )

    def _setup_vector_envs(self, is_dream: bool = False) -> None:
        """Sets up the vector environments for training and testing."""
        env_config = self.config.get_except("environment.base", "name")
        env_funs = [
            make_env(
                self.env_name,
                env_config,
                REC_DIR,
                self.policy_config,
                self.obsnet_config,
                self.intrinsic_config,
                self.is_goal_aware,
                self.factory,
            )
            for _ in range(self.num_envs)
        ]

        if is_dream:
            self.dream_train_envs = ts.env.SubprocVectorEnv(env_funs)
            # no need to create test_envs for dream environment
        else:
            # all the envs created by env_funs() are the same
            self.env = env_funs[0]()
            self.train_envs = ts.env.SubprocVectorEnv(env_funs)
            self.test_envs = ts.env.SubprocVectorEnv(env_funs)

    def _setup_buffers(self, is_dream: bool = False) -> None:
        """Sets up the replay buffers for training and testing."""
        if is_dream:
            self.dream_train_buf = self.factory.create_buffer(
                self.dream_train_buf_size, self.num_dream_envs
            )
            # only test in the real environment, no need to create test_buf
        else:
            self.train_buf = self.factory.create_buffer(
                self.train_buf_size, self.num_envs
            )
            self.test_buf = self.factory.create_buffer(
                self.test_buf_size, self.num_envs
            )

            if self.use_kb:
                self.knowledge_base, self.bandit = (
                    self.factory.create_knowledge_base_and_bandit(
                        self.kb_size, self.num_envs
                    )
                )
            else:
                self.knowledge_base, self.bandit = None, None

    def _setup_networks(self) -> None:
        """Sets up the observation, actor, and critic networks."""
        self.vae, self.mdnrnn = self.factory.create_vae_mdnrnn(
            self.env.observation_space, self.device
        )
        self.obs_net = self.factory.create_obsnet(self.vae.encoder, self.device)
        # hidden_dim * 2 because we concat (h, c) into a single tensor in the policy.forward()
        self.actor_net, self.critic_net = self.factory.create_actor_critic(
            self.obs_net, self.mdnrnn.hidden_dim * 2, self.env.action_space, self.device
        )

    def _setup_models(self) -> None:
        """Sets up the environment model and the self model."""
        self.vae_trainer, self.mdnrnn_trainer = self.factory.create_envmodel_trainers(
            self.vae, self.mdnrnn, self.batch_size, self.learning_rate, self.device
        )
        self.env_model = EnvModel(
            self.vae,
            self.mdnrnn,
            self.vae_trainer,
            self.mdnrnn_trainer,
            device=self.device,
        )

        fast_intrinsic_module, slow_intrinsic_module = (
            self.factory.create_intrinsic_modules(
                self.obs_net,
                self.env.action_space,
                self.train_buf,
                self.batch_size,
                self.learning_rate,
                self.device,
            )
        )
        self.self_model = SelfModel(
            fast_intrinsic_module,
            slow_intrinsic_module,
        )

    def _setup_policy(self) -> None:
        """Sets up the policy."""
        assert self.actor_net.obs_net is self.critic_net.obs_net
        obs_net_params = set(self.actor_net.obs_net.parameters())

        # exclude the obs_net parameters because the obs_net is trained separately
        actor_params = [
            p for p in self.actor_net.parameters() if p not in obs_net_params
        ]
        critic_params = [
            p for p in self.critic_net.parameters() if p not in obs_net_params
        ]
        combined_params = actor_params + critic_params

        self.optimizer = torch.optim.Adam(combined_params, lr=self.learning_rate)
        self.policy = self.factory.create_policy(
            self.self_model,
            self.env_model,
            self.obs_net,
            self.actor_net,
            self.critic_net,
            self.optimizer,
            self.env.action_space,
            self.env.observation_space,
            action_scaling=False,
        )

    def _setup_collectors(self, is_dream: bool = False) -> None:
        """Sets up the collectors for training and testing."""
        if is_dream:
            self.dream_train_collector = self.factory.create_collector(
                self.policy,
                self.dream_train_envs,
                self.dream_train_buf,
                knowledge_base=None,  # the KB only collects from the real env
                bandit=None,
            )
            # only test in the real environment, need set test_collector to None to skip the Trainer's test_step()
            self.dream_test_collector = None
        else:
            self.train_collector = self.factory.create_collector(
                self.policy,
                self.train_envs,
                self.train_buf,
                self.knowledge_base,
                self.bandit,
            )
            self.test_collector = self.factory.create_collector(
                self.policy,
                self.test_envs,
                self.test_buf,
                self.knowledge_base,
                bandit=None,  # do not use the bandit while testing
            )

    def _setup_logger(self) -> None:
        """Sets up the TensorboardLogger."""
        self.log_path = _make_save_path(
            LOG_DIR,
            self.env_name,
            self.policy_config,
            self.obsnet_config,
            self.intrinsic_config,
            self.is_goal_aware,
        )
        self.writer = SummaryWriter(os.path.split(self.log_path)[0])
        self.logger = TensorboardLogger(self.writer)

    def _setup_trainer(self, is_dream: bool = False) -> None:
        """Sets up the trainer."""
        train_collector = (
            self.dream_train_collector if is_dream else self.train_collector
        )
        test_collector = self.dream_test_collector if is_dream else self.test_collector

        trainer = self.factory.create_trainer(
            self.policy,
            train_collector,
            test_collector,
            self.logger,
            is_dream=is_dream,
        )

        attr_name = "dream_trainer" if is_dream else "trainer"
        setattr(self, attr_name, trainer)

    def _setup_dream(self) -> None:
        """Sets up the dream environment and associated components."""
        self.dream_env = DreamEnv(
            self.env_model,
            self.env.observation_space,
            self.env.action_space,
            **self.config.get("environment.dream"),
        )
        self._setup_vector_envs(is_dream=True)
        self._setup_buffers(is_dream=True)
        self._setup_collectors(is_dream=True)
        self._setup_trainer(is_dream=True)

    @contextmanager
    def _enter_dream(self) -> Generator[None, None, None]:
        """Modifies the execution context to enable the agent to act within its dream, restoring the old execution context once dreaming is over."""
        try:
            # __enter__()
            old_fast = self.self_model.fast_intrinsic_module
            old_slow = self.self_model.slow_intrinsic_module
            # disable ICM and HER while dreaming
            self.self_model.fast_intrinsic_module = ZeroICM(
                self.obs_net.o_dim, self.dream_env.action_space.n, 0
            )
            self.self_model.slow_intrinsic_module = ZeroHER(
                self.obs_net, self.dream_train_buf, 0
            )

            self.self_model.slow_intrinsic_module.buf = self.dream_train_buf
            print("\n[+] Dreaming...")
            yield
        finally:
            # __exit__()
            self.self_model.slow_intrinsic_module.buf = self.train_buf

            self.self_model.fast_intrinsic_module = old_fast
            self.self_model.slow_intrinsic_module = old_slow
            print()

    def _run_dream(self) -> None:
        """Runs the dream and updated the correponding epoch stats."""
        with self._enter_dream() as _:
            for dream_epoch_stat in self.dream_trainer:
                self.dream_epoch_stats.append(dream_epoch_stat)

    def _envmodel_is_good(
        self,
        epoch_stat: EpochStats,
        initial_loss_threshold: float = 10.0,
        alpha: float = 0.98,
    ) -> bool:
        """Checks if the environment model is "good", i.e., if it provides an accurate model of the environment.

        Performing this chesk is important because the agent uses the same policy in the dream and the real environment. This means that, if the dream is inaccurate, the agent will act in an inaccurate representation of reality, which might make it quite a bit worse, defeating the whole purpose of dreaming.
        """
        if not hasattr(self, "n_epochs"):
            # keep track of the number of epochs seen
            self.n_epochs = 0
        self.n_epochs += 1

        vae_loss = epoch_stat.training_stat.env_model_stats.vae_loss
        mdnrnn_loss = epoch_stat.training_stat.env_model_stats.mdnrnn_loss

        if not hasattr(self, "vae_loss_ema"):
            self.vae_loss_ema = None
        if not hasattr(self, "mdnrnn_loss_ema"):
            self.mdnrnn_loss_ema = None
        if not hasattr(self, "vae_loss_std_ema"):
            self.vae_loss_std_ema = None
        if not hasattr(self, "mdnrnn_loss_std_ema"):
            self.mdnrnn_loss_std_ema = None
        self.vae_loss_ema = self._update_ema(self.vae_loss_ema, vae_loss.mean)
        self.mdnrnn_loss_ema = self._update_ema(self.mdnrnn_loss_ema, mdnrnn_loss.mean)
        self.vae_loss_std_ema = self._update_ema(self.vae_loss_std_ema, vae_loss.std)
        self.mdnrnn_loss_std_ema = self._update_ema(
            self.mdnrnn_loss_std_ema, mdnrnn_loss.std
        )

        mean_loss = (self.vae_loss_ema + self.mdnrnn_loss_ema) / 2
        mean_std = (self.vae_loss_std_ema + self.mdnrnn_loss_std_ema) / 2

        # adaptive threshold based on moving averages
        loss_threshold = initial_loss_threshold * alpha**self.n_epochs
        if mean_loss > loss_threshold or mean_std > 0.05 * mean_loss:
            return False

        return True

    def _update_ema(self, ema_value, new_value, alpha=0.1):
        """Updates the exponential moving average."""
        if ema_value is None:
            return new_value
        return alpha * new_value + (1 - alpha) * ema_value

    def _plot(self, save_pdf: bool = True) -> None:
        """Plots the data.

        Its default behaviour is to save the plot to a PDF file to not block execution.
        """
        plot_path = _make_save_path(
            PLOT_DIR,
            self.env_name,
            self.policy_config,
            self.obsnet_config,
            self.intrinsic_config,
            self.is_goal_aware,
        )

        plotter = self.factory.create_plotter(self.epoch_stats)
        # if save_pdf is False, the path will be ignored
        plotter.plot(figsize=(20, 16), save_pdf=save_pdf, pdf_path=plot_path)

    def _record_rollout(self) -> None:
        """Records a rollout lasting one episode."""
        obs, info = self.env.reset()
        done = False

        while not done:
            action = self.policy(
                ts.data.Batch(obs=np.array([obs]), info=np.array([info]))
            ).act.item()
            obs, info, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

        self.env.close()


def make_env(
    env_name: str,
    env_config: str,
    rec_dir: str,
    policy_config: str,
    obsnet_config: str,
    intrinsic_config: str,
    is_goal_aware: bool,
    factory: ExperimentFactory,
):
    """Creates a single environment.

    We need to create envs this way to ensure picklability, which is necessary for Tianshou's SubprocVectorEnv.
    """

    def _thunk():
        env = gym.make(env_name, **env_config)
        rec_path = _make_save_path(
            rec_dir,
            env_name,
            policy_config,
            obsnet_config,
            intrinsic_config,
            is_goal_aware,
            ext="mp4" if env.render_mode == "rgb_array" else "ttyrec",
        )
        env = factory.wrap_env(env, rec_path)
        return env

    return _thunk


def _make_save_path(
    base_path: str,
    env_config: str,
    policy_config: str,
    obsnet_config: str,
    intrinsic_config: str,
    is_goal_aware: bool,
    ext: str | None = None,
) -> str:
    """Creates a path to save the artefacts (plots, recordings, and logs)."""
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    save_path = os.path.join(
        base_path,
        env_config.lower(),
        policy_config.lower(),
        obsnet_config.lower(),
        intrinsic_config.lower(),
        "goal" if is_goal_aware else "vanilla",
        f"{timestamp}.{ext}" if ext else timestamp,
    )
    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path

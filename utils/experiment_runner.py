from contextlib import contextmanager
from datetime import datetime
import os

import torch
import gymnasium as gym
import tianshou as ts
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models import SelfModel, EnvModel
from config import ConfigManager
from .experiment_factory import ExperimentFactory

from environments import DreamEnv

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
        device: torch.device,
    ):
        self.base_config_path = base_config_path
        self.env_config = env_config
        self.policy_config = policy_config
        self.obsnet_config = obsnet_config
        self.intrinsic_config = intrinsic_config
        self.model_config = model_config
        self.device = device

        self._setup_config()
        self.env_name = self.config.get("environment.base.name")
        self.num_train_envs = self.config.get("environment.vec.num_train_envs")
        self.dream_num_train_envs = self.config.get(
            "environment.vec.dream_num_train_envs"
        )
        self.num_test_envs = self.config.get("environment.vec.num_test_envs")

        self.train_buf_size = self.config.get("buffers.train_buf_size")
        self.dream_train_buf_size = self.config.get("buffers.dream_train_buf_size")
        self.test_buf_size = self.config.get("buffers.test_buf_size")

        # use the real batch size by default
        self.batch_size = self.config.get("training.real.batch_size")
        self.learning_rate = self.config.get("policy.learning_rate")

        self.factory = ExperimentFactory(self.config)
        self.is_goal_aware = self.factory.is_goal_aware

    def setup(self):
        print("[+] Setting up the environments...")
        self._setup_environment()
        self._setup_vector_envs(self.env, is_dream=False)
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

    def run(self, save_pdf_plot: bool = True):
        """Runs the experiment and collects epoch statistics."""
        print("\n[+] Running the experiment...")
        self.epoch_stats, self.dream_epoch_stats = [], []
        for epoch_stat in self.trainer:
            self.epoch_stats.append(epoch_stat)

            # TODO this is possibly the simplest approach we could take (making it more complex is mostly a triviality, though)
            with self._dream_buffer() as _:
                for dream_epoch_stat in self.dream_trainer:
                    self.dream_epoch_stats.append(dream_epoch_stat)

        print("\n[+] Plotting..." if not save_pdf_plot else "\n[+] Saving the plot...")
        self._plot(save_pdf=save_pdf_plot)

        print("[+] Recording a rollout...")
        self._record_rollout()

        print("[+] All done!")

    def _setup_config(self):
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

    def _setup_environment(self):
        """Sets up the gym environment."""
        try:
            self.env = gym.make(
                self.env_name, **self.config.get_except("environment.base", "name")
            )
            self.rec_path = self._make_save_path(
                REC_DIR,
                self.env_name,
                self.policy_config,
                self.obsnet_config,
                self.intrinsic_config,
                self.is_goal_aware,
                ext="mp4" if self.env.render_mode == "rgb_array" else "ttyrec",
            )
            self.env = self.factory.wrap_env(self.env, self.rec_path)
        except gym.error.Error as e:
            raise RuntimeError(f"Failed to create environment {self.env_name}: {e}")

    def _setup_vector_envs(self, environment: gym.Env, is_dream: bool = False):
        """Sets up the vector environments for training and testing."""
        if is_dream:
            self.dream_train_envs = ts.env.DummyVectorEnv(
                [lambda: environment for _ in range(self.dream_num_train_envs)]
            )
            # only test in the real environment, no need to create test_envs
        else:
            self.train_envs = ts.env.DummyVectorEnv(
                [lambda: environment for _ in range(self.num_train_envs)]
            )
            self.test_envs = ts.env.DummyVectorEnv(
                [lambda: environment for _ in range(self.num_test_envs)]
            )

    def _setup_buffers(self, is_dream: bool = False):
        """Sets up the replay buffers for training and testing."""
        if is_dream:
            self.dream_train_buf = self.factory.create_buffer(
                self.dream_train_buf_size, self.dream_num_train_envs
            )
            # only test in the real environment, no need to create test_buf
        else:
            self.train_buf = self.factory.create_buffer(
                self.train_buf_size, self.num_train_envs
            )
            self.test_buf = self.factory.create_buffer(
                self.test_buf_size, self.num_test_envs
            )

    def _setup_networks(self):
        """Sets up the observation, actor, and critic networks."""
        self.vae, self.mdnrnn = self.factory.create_vae_mdnrnn(
            self.env.observation_space, self.device
        )
        self.obs_net = self.factory.create_obsnet(self.vae.encoder, self.device)
        self.actor_net, self.critic_net = self.factory.create_actor_critic(
            self.obs_net, self.env.action_space, self.device
        )

    def _setup_models(self):
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
            self.obs_net,
            fast_intrinsic_module,
            slow_intrinsic_module,
            device=self.device,
        )

    def _setup_policy(self):
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
            self.actor_net,
            self.critic_net,
            self.optimizer,
            self.env.action_space,
            self.env.observation_space,
            False,  # no action scaling
        )

    def _setup_collectors(self, is_dream: bool = False):
        """Sets up the collectors for training and testing."""
        if is_dream:
            self.dream_train_collector = self.factory.create_collector(
                self.policy, self.dream_train_envs, self.dream_train_buf
            )
            # only test in the real environment, need set test_collector to None to skip the Trainer's test_step()
            self.dream_test_collector = None
        else:
            self.train_collector = self.factory.create_collector(
                self.policy, self.train_envs, self.train_buf
            )
            self.test_collector = self.factory.create_collector(
                self.policy, self.test_envs, self.test_buf
            )

    def _setup_logger(self):
        """Sets up the TensorboardLogger."""
        self.log_path = self._make_save_path(
            LOG_DIR,
            self.env_name,
            self.policy_config,
            self.obsnet_config,
            self.intrinsic_config,
            self.is_goal_aware,
        )
        self.writer = SummaryWriter(os.path.split(self.log_path)[0])
        self.logger = TensorboardLogger(self.writer)

    def _setup_trainer(self, is_dream: bool = False):
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

    def _setup_dream(self):
        """Sets up the dream environment and associated components."""
        self.dream_env = DreamEnv(
            self.env_model, self.env.observation_space, self.env.action_space
        )
        self._setup_vector_envs(self.dream_env, is_dream=True)
        self._setup_buffers(is_dream=True)

        self._setup_collectors(is_dream=True)
        self._setup_trainer(is_dream=True)

    @contextmanager
    def _dream_buffer(self):
        """Changes the SelfModel buffer to point to the dream buffer and switches it back before returning to the real environment."""
        try:
            print("\n[+] Dreaming...")
            self.self_model.slow_intrinsic_module.buf = self.dream_train_buf
            yield
        finally:
            self.self_model.slow_intrinsic_module.buf = self.train_buf
            print()

    def _plot(self, save_pdf: bool = True):
        """Plots the data.

        Its default behaviour is to save the plot to a PDF file to not block execution.
        """
        plot_path = self._make_save_path(
            PLOT_DIR,
            self.env_name,
            self.policy_config,
            self.obsnet_config,
            self.intrinsic_config,
            self.is_goal_aware,
        )

        plotter = self.factory.create_plotter(self.epoch_stats)
        # if save_pdf is False, the path will be ignored
        plotter.plot(figsize=(12, 8), save_pdf=save_pdf, pdf_path=plot_path)

    def _record_rollout(self):
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

    def _make_save_path(
        self,
        base_path: str,
        env_config: str,
        policy_config: str,
        obsnet_config: str,
        intrinsic_config: str,
        is_goal_aware: bool,
        ext: str | None = None,
    ):
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

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

# TODO test code
# from environments import DreamEnv

ART_DIR = "artefacts"
LOG_DIR = f"{ART_DIR}/logs"
PLOT_DIR = f"{ART_DIR}/plots"
REC_DIR = f"{ART_DIR}/recs"


class ExperimentRunner:
    def __init__(
        self,
        base_config_path,
        env_config,
        policy_config,
        obsnet_config,
        intrinsic_config,
        model_config,
        device,
    ):
        self.base_config_path = base_config_path
        self.env_config = env_config
        self.policy_config = policy_config
        self.obsnet_config = obsnet_config
        self.intrinsic_config = intrinsic_config
        self.model_config = model_config
        self.device = device

        self._setup_config()
        self.factory = ExperimentFactory(self.config)
        self.is_goal_aware = self.factory.is_goal_aware

        print("[+] Setting up the environments...")
        self._setup_environment()
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
        self._setup_vector_envs()

        print("[+] Setting up the buffers...")
        self._setup_buffers()

        print("[+] Setting up the networks...")
        self._setup_networks()

        print("[+] Setting up the models...")
        self._setup_models()

        print("[+] Setting up the policy...")
        self._setup_policy()

        print("[+] Setting up the collector...")
        self._setup_collectors()

        print("[+] Setting up the logger...")
        self._setup_logger()

        print("[+] Setting up the trainer...")
        self._setup_trainer()

    def run(self):
        """Runs the experiment and collects epoch statistics."""
        print("\n[+] Running the experiment...")
        self.epoch_stats = []
        for epoch_stat in self.trainer:
            self.epoch_stats.append(epoch_stat)

        save_pdf = True
        print("\n[+] Plotting..." if not save_pdf else "\n[+] Saving the plot...")
        self._plot(save_pdf=save_pdf)

        print("[+] Recording a rollout...")
        self._record_rollout()

        print("[+] All done!")

    def _setup_config(self):
        """Sets up and validates the configuration."""
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
        self.env_name = self.config.get("environment.base.name")
        try:
            self.env = gym.make(
                self.env_name, **self.config.get_except("environment.base", "name")
            )
        except gym.error.Error as e:
            raise RuntimeError(f"Failed to create environment {self.env_name}: {e}")

    def _setup_vector_envs(self):
        """Sets up the vector environments for training and testing."""
        num_train_envs = self.config.get("environment.vec.num_train_envs")
        num_test_envs = self.config.get("environment.vec.num_test_envs")
        self.train_envs = ts.env.DummyVectorEnv(
            [lambda: self.env for _ in range(num_train_envs)]
        )
        self.test_envs = ts.env.DummyVectorEnv(
            [lambda: self.env for _ in range(num_test_envs)]
        )

    def _setup_buffers(self):
        """Sets up the replay buffers for training and testing."""
        train_buf_size = self.config.get("buffers.train_buf_size")
        test_buf_size = self.config.get("buffers.test_buf_size")
        self.train_buf = self.factory.create_buffer(
            train_buf_size, len(self.train_envs)
        )
        self.test_buf = self.factory.create_buffer(test_buf_size, len(self.test_envs))

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
        batch_size = self.config.get("training.batch_size")
        learning_rate = self.config.get("policy.learning_rate")

        self.vae_trainer, self.mdnrnn_trainer = self.factory.create_envmodel_trainers(
            self.vae, self.mdnrnn, batch_size, learning_rate, self.device
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
                batch_size,
                learning_rate,
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
        lr = self.config.get("policy.learning_rate")
        combined_params = set(
            list(self.actor_net.parameters()) + list(self.critic_net.parameters())
        )
        self.optimizer = torch.optim.Adam(combined_params, lr=lr)
        self.policy = self.factory.create_policy(
            self.self_model,
            self.env_model,
            self.actor_net,
            self.critic_net,
            self.optimizer,
            self.env.action_space,
            self.env.observation_space,
            False,  # No action scaling
        )

    def _setup_collectors(self):
        """Sets up the collectors for training and testing."""
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

    def _setup_trainer(self):
        """Sets up the trainer."""
        self.trainer = self.factory.create_trainer(
            self.policy, self.train_collector, self.test_collector, self.logger
        )

    def _plot(self, save_pdf=True):
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
        base_path,
        env_config,
        policy_config,
        obsnet_config,
        intrinsic_config,
        is_goal_aware,
        ext=None,
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
        # make the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

from typing import Any, Dict
from core.types import (
    GoalBatchProtocol,
    LatentObsActNextBatchProtocol,
    FastIntrinsicModuleProtocol,
    SlowIntrinsicModuleProtocol,
)

from tianshou.policy.base import TrainingStats
import numpy as np
import torch
import logging
from dataclasses import dataclass

from language import SentimentAnalyser


@dataclass
class GoalStats:
    """Statistics about goal selection strategy."""

    goal_strategy: str
    avg_steps_per_goal: float = 0.0
    total_goal_selections: int = 0
    total_goal_resets: int = 0
    active_goals: int = 0


class SelfModel:
    """The SelfModel represents an agent's model of itself.

    It is, fundamentally, a container for all things that should happen exclusively within an agent, independently of the outside world.
    """

    def __init__(
        self,
        fast_intrinsic_module: FastIntrinsicModuleProtocol,
        slow_intrinsic_module: SlowIntrinsicModuleProtocol,
        use_sentiment: bool = False,
        goal_strategy: str = "random",
        noise_scale: float = 0.5,
        noise_seed: int = 42,
        log_goals: bool = False,
    ) -> None:
        self.fast_intrinsic_module = fast_intrinsic_module
        self.slow_intrinsic_module = slow_intrinsic_module
        # sentiment analysis provides extra reward at collect time (added to the fast intrinsic reward)
        if use_sentiment:
            self.sentiment_analyser = SentimentAnalyser()
        else:
            self.sentiment_analyser = None

        # Store goal selection parameters
        self.goal_strategy = goal_strategy
        self.noise_scale = noise_scale
        self.noise_seed = noise_seed

        # to store environment-specific goals
        self.env_goals: Dict[int, np.ndarray] = {}

        # for tracking goal consistency
        self.log_goals = log_goals
        if self.log_goals:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("GoalConsistency")
            self.goal_selections = 0
            self.goal_resets = 0

        self._setup_goal_strategy()

    @torch.no_grad()
    def select_goal(self, latent_obs: torch.Tensor) -> np.ndarray:
        """Selects a goal for the agent to pursue based on the batch of observations it receives in input.

        Goals remain consistent for each environment until explicitly reset (by the policy).
        """
        latent_obs_np = latent_obs.cpu().numpy().astype(np.float32)
        goals = np.zeros_like(latent_obs_np)

        for i in range(latent_obs_np.shape[0]):
            env_key = i
            # if environment doesn't have a goal yet, generate one
            if env_key not in self.env_goals:
                self.env_goals[env_key] = self._select_goal_fn(latent_obs_np, env_key)

                if self.log_goals:
                    self.goal_selections += 1
                    self.logger.info(
                        f"New goal created for env {env_key} using {self.goal_strategy} strategy"
                    )

            goals[i] = self.env_goals[env_key]

        # Log statistics periodically
        if (
            self.log_goals
            and self.goal_selections % 100 == 0
            and self.goal_selections > 0
        ):
            self.logger.info(
                f"Goal stats - Selections: {self.goal_selections}, Resets: {self.goal_resets}"
            )
            if self.goal_resets > 0:
                self.logger.info(
                    f"Average steps per goal: {self.goal_selections / self.goal_resets:.2f}"
                )

        return goals

    @torch.no_grad()
    def fast_intrinsic_reward(self, batch: LatentObsActNextBatchProtocol) -> np.ndarray:
        """A fast system for computing intrinsic motivation, inspired by the dual process theory (https://en.wikipedia.org/wiki/Dual_process_theory).

        This intrinsic computation happens at collect time, and is somewhat conceptually analogous to Kahneman's System 1.
        """
        return self.fast_intrinsic_module.get_reward(batch)

    @torch.no_grad()
    def sentiment_reward(self, message: str) -> np.ndarray:
        """A reward based on the sentiment of the messages provided by the environment."""
        if self.sentiment_analyser is not None:
            return self.sentiment_analyser.get_reward(message)
        else:
            return np.zeros(1)

    @torch.no_grad()
    def slow_intrinsic_reward_(self, indices: np.ndarray) -> np.ndarray:
        """A slow system for computing intrinsic motivation, inspired by the dual process theory (https://en.wikipedia.org/wiki/Dual_process_theory).

        This intrinsic computation happens at update time, and is somewhat conceptually analogous to Kahneman's System 2.
        """
        # we cannot return the reward here because modifying the buffer requires access to its internals
        self.slow_intrinsic_module.rewrite_rewards_(indices)

    def learn(self, batch: GoalBatchProtocol, **kwargs: Any) -> TrainingStats:
        stats = self.fast_intrinsic_module.learn(batch, **kwargs)
        stats.goal_stats = self._get_goal_stats()
        return stats

    def reset_env_goals(self, env_ids: np.ndarray) -> None:
        """Reset goals for specific environments."""
        for env_id in env_ids:
            if env_id in self.env_goals:
                if self.log_goals:
                    self.logger.info(f"Resetting goal for env {env_id}")
                    self.goal_resets += 1

                del self.env_goals[env_id]

    def _setup_goal_strategy(self) -> None:
        """Configure the goal selection function based on the specified strategy."""
        if self.goal_strategy == "zero":
            self._select_goal_fn = self._zero_goal_strategy
        elif self.goal_strategy == "random":
            self._select_goal_fn = self._random_goal_strategy
        else:
            raise ValueError(f"Unknown goal strategy: {self.goal_strategy}")

        if self.log_goals:
            self.logger.info(f"Using goal strategy: {self.goal_strategy}")

    def _zero_goal_strategy(
        self, latent_obs_np: np.ndarray, env_key: int
    ) -> np.ndarray:
        """Zero goal strategy - goals are all zeros."""
        return np.zeros_like(latent_obs_np[env_key])

    def _random_goal_strategy(
        self, latent_obs_np: np.ndarray, env_key: int
    ) -> np.ndarray:
        """Random goal strategy - goals are current observation plus random noise."""
        rng = np.random.RandomState(self.noise_seed + env_key)
        noise = rng.normal(0, self.noise_scale, size=latent_obs_np[env_key].shape)
        return latent_obs_np[env_key] + noise

    def _get_goal_stats(self) -> GoalStats:
        """Return statistics about goal generation."""
        avg_steps = 0.0
        if self.log_goals and self.goal_resets > 0:
            avg_steps = self.goal_selections / self.goal_resets

        return GoalStats(
            goal_strategy=self.goal_strategy,
            avg_steps_per_goal=avg_steps,
            total_goal_selections=getattr(self, "goal_selections", 0),
            total_goal_resets=getattr(self, "goal_resets", 0),
            active_goals=len(self.env_goals),
        )

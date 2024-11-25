from .utils import is_similar


class TrajectoryBandit:
    """A bandit which selects which trajectory from the knowledge base the agent should use."""

    # TODO

    def __init__(self):
        self.arms = {}  # map trajectory indices to bandit values

    def select_trajectory(self, state, trajectories):
        # find matching trajectories
        matching_indices = []
        for idx, traj in enumerate(trajectories):
            if is_similar(state, traj[0].latent_obs):
                matching_indices.append(idx)
        if not matching_indices:
            return None

        # select trajectory using bandit algorithm
        selected_idx = bandit_algorithm(matching_indices, self.arms)
        return trajectories[selected_idx]

    def update_bandit(self, trajectory_idx, reward):
        # TODO update the bandit values based on the received reward
        pass


def bandit_algorithm(matching_indices, arms):
    # TODO bandit algorithm
    pass

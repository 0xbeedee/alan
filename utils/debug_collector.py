import time
from core import GoalCollector


class DebugCollector(GoalCollector):
    """The DebugCollector wraps the Collector in a series of informative prints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_count = 0
        self.total_transitions = 0

    def collect(self, n_step=None, n_episode=None, random=False, render=None):
        start_time = time.time()
        result = super().collect(n_step, n_episode, random, render)
        end_time = time.time()

        self.collection_count += 1
        transitions_collected = result.n_collected_steps
        self.total_transitions += transitions_collected

        print(f"Collection {self.collection_count}:")
        print(f"  Time taken: {end_time - start_time:.4f}s")
        print(f"  Transitions collected: {transitions_collected}")
        print(f"  Total transitions so far: {self.total_transitions}")
        print(f"  Buffer size: {len(self.buffer)}")
        print(f"  n_step requested: {n_step}")
        print(f"  n_episode requested: {n_episode}")
        print(f"  Random: {random}")

        return result

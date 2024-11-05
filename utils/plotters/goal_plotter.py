from typing import List, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from .base_plotter import BasePlotter, PlotFunctionTuple


class GoalStatsPlotter(BasePlotter):
    def _get_plot_functions(self) -> Sequence[PlotFunctionTuple]:
        return [
            (self._plot_returns, ("returns",)),
            (self._plot_returns, ("int_returns",)),
            (self._plot_losses, ("policy_stats",)),
            (self._plot_losses, ("self_model_stats",)),
            (self._plot_losses, ("env_model_stats",)),
            (self._plot_intra_episodic_returns, ("train",)),
            (self._plot_intra_episodic_returns, ("test",)),
        ]

    def _plot_losses(self, ax: plt.Axes, loss_type: str) -> None:
        losses = self._extract_losses(loss_type)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(losses)))

        for (loss_name, loss_values), color in zip(losses.items(), colors):
            stds = np.zeros_like(loss_values)  # for backwards compatibility
            self._plot_with_ci(ax, self.epochs, loss_values, stds, loss_name, color)

        ax.set_title(f"{loss_type.replace('_', ' ').title()[:-6]} Losses")
        ax.set_ylabel("Loss")
        ax.legend()
        self._set_consistent_x_axis(ax)

    def _extract_losses(self, loss_type: str) -> Dict[str, List[float]]:
        losses = defaultdict(list)
        for stats in self.epoch_stats:
            training_stat = self._get_nested_attr(stats, ["training_stat"])
            loss_object = self._get_nested_attr(training_stat, [loss_type])
            loss_dict = loss_object.get_loss_stats_dict()
            for loss, value in loss_dict.items():
                losses[loss].append(value)
        return losses

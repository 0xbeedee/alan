from typing import List, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from .base_plotter import BasePlotter, PlotFunctionTuple


class VanillaStatsPlotter(BasePlotter):
    # TODO this only works with PPO (and PPO-like policies) for now
    def _get_plot_functions(self) -> Sequence[PlotFunctionTuple]:
        return [
            (self._plot_returns, ("returns",)),
            (self._plot_empty, ("Intrinsic Returns (N/A)",)),
            (self._plot_losses, ()),
            (self._plot_empty, ("Self Model Stats (N/A)",)),
            (self._plot_intra_episodic_returns, ("train",)),
            (self._plot_intra_episodic_returns, ("test",)),
        ]

    def _plot_losses(self, ax):
        losses = self._extract_vanilla_losses()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(losses)))

        for (loss_name, loss_values), color in zip(losses.items(), colors):
            stds = np.zeros_like(loss_values)  # for backwards compatibility
            self._plot_with_ci(ax, self.epochs, loss_values, stds, loss_name, color)

        ax.set_title("Training Losses")
        ax.set_ylabel("Loss")
        ax.legend()
        self._set_consistent_x_axis(ax)

    def _extract_vanilla_losses(self) -> Dict[str, List[float]]:
        losses = defaultdict(list)
        for stats in self.epoch_stats:
            training_stat = stats.training_stat
            for loss_name in ["loss", "clip_loss", "vf_loss", "ent_loss"]:
                loss_data = getattr(training_stat, loss_name)
                losses[loss_name].append(loss_data.mean)
        return losses

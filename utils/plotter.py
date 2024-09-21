from typing import List, Dict, Tuple, Any, Optional
from tianshou.data import EpochStats

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict


class EpochStatsPlotter:
    def __init__(self, epoch_stats: List[EpochStats]):
        self.epoch_stats = epoch_stats
        self.epochs = [stats.epoch for stats in epoch_stats]

    def plot(
        self,
        figsize: Tuple[int, int] = (15, 18),
        save_pdf: bool = False,
        pdf_path: Optional[str] = None,
    ) -> None:
        self._set_plot_style()
        fig, axs = plt.subplots(3, 2, figsize=figsize)

        plot_functions = [
            (self._plot_returns, ("returns",)),
            (self._plot_returns, ("int_returns",)),
            (self._plot_losses, ("policy_stats",)),
            (self._plot_losses, ("self_model_stats",)),
            (self._plot_intra_episodic_returns, ("train",)),
            (self._plot_intra_episodic_returns, ("test",)),
        ]

        for (i, j), (func, args) in zip(np.ndindex(3, 2), plot_functions):
            func(axs[i, j], *args)

        self._finalize_plot(fig, axs, save_pdf, pdf_path)
        plt.show()

    def _set_plot_style(self):
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
                "font.size": 10,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "figure.titlesize": 16,
            }
        )

    def _plot_returns(self, ax, returns_type: str):
        ax2 = ax.twinx()
        colours = sns.color_palette("colorblind")

        for idx, (collect_type, ax_) in enumerate([("test", ax), ("train", ax2)]):
            returns, returns_std = self._extract_data(
                [f"{collect_type}_collect_stat", f"{returns_type}_stat"]
            )
            self._plot_with_ci(
                ax_,
                self.epochs,
                returns,
                returns_std,
                f"{collect_type} returns",
                colours[idx],
            )

        ax.set_title("Returns" if returns_type == "returns" else "Intrinsic Returns")
        ax.set_ylabel("Test")
        ax2.set_ylabel("Train")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        self._set_consistent_x_axis(ax)

    def _plot_losses(self, ax, loss_type: str):
        losses = self._extract_losses(loss_type)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(losses)))

        for (loss_name, loss_values), color in zip(losses.items(), colors):
            stds = np.zeros_like(loss_values)  # for backwards compatibility
            self._plot_with_ci(ax, self.epochs, loss_values, stds, loss_name, color)

        ax.set_title(f"{loss_type.replace('_', ' ').title()} Losses")
        ax.set_ylabel("Loss")
        ax.legend()
        self._set_consistent_x_axis(ax)

    def _plot_intra_episodic_returns(self, ax, data_type: str):
        """Plots the returns achieved within an episode using boxplots."""
        intra_returns = [
            self._get_nested_attr(stats, [f"{data_type}_collect_stat", "returns"])
            for stats in self.epoch_stats
        ]

        box_positions = np.arange(1, len(self.epochs) + 1)
        ax.boxplot(intra_returns, positions=box_positions)
        ax.set_title(f"{data_type.capitalize()} Intra-episodic Returns")
        ax.set_ylabel("Returns")
        self._set_consistent_x_axis(ax)

    def _plot_with_ci(self, ax, x, y, yerr, label, color):
        """Adds confidence intervals to the provided data."""
        ax.plot(x, y, label=label, color=color, linewidth=2)
        ax.fill_between(
            x,
            np.array(y) - np.array(yerr),
            np.array(y) + np.array(yerr),
            color=color,
            alpha=0.3,
        )

    def _set_consistent_x_axis(self, ax):
        ax.set_xticks(self.epochs)
        ax.set_xlim(min(self.epochs) - 1, max(self.epochs) + 1)
        ax.grid(True, axis="x")

    def _extract_data(self, key_path: List[str]) -> Tuple[List[float], List[float]]:
        means, stds = [], []
        for stats in self.epoch_stats:
            data = self._get_nested_attr(stats, key_path)
            if isinstance(data, dict):
                means.append(data.get("mean", 0))
                stds.append(data.get("std", 0))
            elif hasattr(data, "mean") and hasattr(data, "std"):
                means.append(data.mean)
                stds.append(data.std)
            else:
                means.append(data)
                stds.append(0)
        return means, stds

    def _extract_losses(self, loss_type: str) -> Dict[str, List[float]]:
        losses = defaultdict(list)
        for stats in self.epoch_stats:
            training_stat = self._get_nested_attr(stats, ["training_stat"])
            loss_object = self._get_nested_attr(training_stat, [loss_type])
            loss_dict = loss_object.get_loss_stats_dict()
            for loss, value in loss_dict.items():
                losses[loss].append(value)
        return losses

    def _get_nested_attr(self, obj: Any, attr_path: List[str]) -> Any:
        for attr in attr_path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            elif isinstance(obj, dict) and attr in obj:
                obj = obj[attr]
            else:
                return None
        return obj

    def _finalize_plot(self, fig, axs, save_pdf: bool, pdf_path: Optional[str]):
        for ax in axs[2, :]:
            ax.set_xlabel("Epoch")
        for ax in axs[:-1, :].flatten():
            ax.tick_params(labelbottom=False)

        plt.tight_layout()
        fig.subplots_adjust(top=1, hspace=0.2, wspace=0.25)

        if save_pdf:
            if pdf_path is None:
                raise ValueError("pdf_path must be provided when save_pdf is True")
            with PdfPages(f"{pdf_path}.pdf") as pdf:
                pdf.savefig(fig, bbox_inches="tight")

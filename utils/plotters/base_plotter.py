from typing import List, Sequence, Tuple, Any, Optional, Callable, TypeVar
from math import ceil, sqrt

from tianshou.data import EpochStats
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# a type var to denote the various arguments a plot function might take
PlotArgs = TypeVar("PlotArgs")
PlotFunction = Callable[[plt.Axes, PlotArgs], None]
PlotFunctionTuple = Tuple[PlotFunction, Tuple[Any, ...]]


class BasePlotter(ABC):
    def __init__(self, epoch_stats: List[EpochStats]) -> None:
        self.epoch_stats = epoch_stats
        self.epochs = [stats.epoch for stats in epoch_stats]

    def plot(
        self,
        figsize: Tuple[int, int] = (15, 18),
        save_pdf: bool = False,
        pdf_path: Optional[str] = None,
        ncols: Optional[int] = None,
    ) -> None:
        self._set_plot_style()
        plot_functions = self._get_plot_functions()
        num_plots = len(plot_functions)

        if ncols is None:
            # determine number of columns for a nearly square layout
            ncols = ceil(sqrt(num_plots))
        ncols = max(1, ncols)  # ncols must be >= 1
        nrows = ceil(num_plots / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axs_flat = axs.flatten()
        for ax, (func, args) in zip(axs_flat, plot_functions):
            func(ax, *args)
        # hide any remaining axes if there are more subplots than plots
        for ax in axs_flat[num_plots:]:
            ax.axis("off")

        self._finalize_plot(fig, axs, save_pdf, pdf_path)
        if not save_pdf:
            plt.show()

    @abstractmethod
    def _get_plot_functions(self) -> Sequence[PlotFunctionTuple]:
        """Returns the plot functions to use, i.e., which plots the figure should contain."""

    def _set_plot_style(self) -> None:
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

    def _plot_returns(self, ax: plt.Axes, returns_type: str) -> None:
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
                f"{collect_type}",
                colours[idx],
            )

        ax.set_title(
            "Mean "
            + ("Extrinsic" if returns_type == "returns" else "Intrinsic")
            + " Returns"
        )
        ax.set_ylabel("Test Returns")
        ax2.set_ylabel("Train Returns")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        self._set_consistent_x_axis(ax)

    def _plot_intra_episodic_returns(self, ax: plt.Axes, data_type: str) -> None:
        """Plots the returns achieved within an episode using boxplots."""
        intra_returns = [
            self._get_nested_attr(stats, [f"{data_type}_collect_stat", "returns"])
            for stats in self.epoch_stats
        ]

        box_positions = np.arange(1, len(self.epochs) + 1)
        ax.boxplot(intra_returns, positions=box_positions)
        ax.set_title(f"Intra-epoch {data_type.capitalize()} Returns")
        ax.set_ylabel("Returns")
        self._set_consistent_x_axis(ax)

    def _plot_with_ci(
        self,
        ax: plt.Axes,
        x: Sequence[int],
        y: Sequence[float],
        yerr: Sequence[float],
        label: str,
        color: Any,
    ) -> None:
        """Adds confidence intervals to the provided data."""
        ax.plot(x, y, label=label, color=color, linewidth=2)
        ax.fill_between(
            x,
            np.array(y) - np.array(yerr),
            np.array(y) + np.array(yerr),
            color=color,
            alpha=0.3,
        )

    def _plot_empty(self, ax: plt.Axes, title: str) -> None:
        ax.text(0.5, 0.5, title, ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    def _set_consistent_x_axis(self, ax: plt.Axes) -> None:
        ax.set_xlabel("Epochs")
        ax.set_xticks(self.epochs)
        ax.set_xlim(min(self.epochs) - 1, max(self.epochs) + 1)
        ax.grid(True, axis="x")

    def _extract_data(self, key_path: List[str]) -> Tuple[List[float], List[float]]:
        means = []
        stds = []
        for stats in self.epoch_stats:
            data: Any = self._get_nested_attr(stats, key_path)
            if isinstance(data, dict):
                means.append(data.get("mean", 0.0))
                stds.append(data.get("std", 0.0))
            elif hasattr(data, "mean") and hasattr(data, "std"):
                means.append(data.mean)
                stds.append(data.std)
            else:
                means.append(float(data))
                stds.append(0.0)
        return means, stds

    def _get_nested_attr(self, obj: Any, attr_path: List[str]) -> Any:
        for attr in attr_path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            elif isinstance(obj, dict) and attr in obj:
                obj = obj[attr]
            else:
                return None
        return obj

    def _finalize_plot(
        self,
        fig: plt.Figure,
        axs: np.ndarray,
        save_pdf: bool,
        pdf_path: Optional[str],
    ) -> None:
        plt.tight_layout()
        fig.subplots_adjust(top=1, hspace=0.3, wspace=0.3)

        if save_pdf:
            if pdf_path is None:
                raise ValueError("pdf_path must be provided when save_pdf is True")
            with PdfPages(f"{pdf_path}.pdf") as pdf:
                pdf.savefig(fig, bbox_inches="tight")

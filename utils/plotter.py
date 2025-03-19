from typing import List, Sequence, Tuple, Any, Optional, Callable, TypeVar, Dict


from tianshou.data import EpochStats

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# a type var to denote the various arguments a plot function might take
PlotArgs = TypeVar("PlotArgs")
PlotFunction = Callable[[plt.Axes, PlotArgs], None]
PlotFunctionTuple = Tuple[PlotFunction, Tuple[Any, ...]]


class Plotter:
    def __init__(self, epoch_stats: List[EpochStats]) -> None:
        self.epoch_stats = epoch_stats
        self.epochs = [stats.epoch for stats in epoch_stats]

    def plot(
        self,
        figsize: Tuple[int, int] = (18, 14),
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

    def _set_plot_style(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
                "font.size": 14,
                "axes.labelsize": 12,
                "axes.titlesize": 16,
                "figure.titlesize": 18,
            }
        )

    def _get_plot_functions(self) -> Sequence[PlotFunctionTuple]:
        functions = [
            (self._plot_returns, ("returns",)),
            (self._plot_returns, ("int_returns",)),
            (self._plot_intra_episodic_returns, ("train",)),
            (self._plot_policy_losses, ()),
            (self._plot_selfmodel_losses, ()),
            (self._plot_envmodel_losses, ()),
            # we only care about episodic returns in the test set
            # (self._plot_intra_episodic_returns, ("test",)),
        ]
        # if self._has_goal_stats():
        #     functions.append((self._plot_goal_strategy_stats, ()))
        return functions

    def _plot_returns(self, ax: plt.Axes, returns_type: str) -> None:
        ax2 = ax.twinx()
        colours = sns.color_palette("colorblind")

        for idx, (collect_type, ax_) in enumerate([("test", ax), ("train", ax2)]):
            returns, returns_std = self._extract_data(
                [f"{collect_type}_collect_stat", f"{returns_type}_stat"]
            )
            collect_type = collect_type.replace("_", " ")
            self._plot_with_ci(
                ax_,
                self.epochs,
                returns,
                returns_std,
                collect_type,
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

    def _plot_policy_losses(self, ax: plt.Axes) -> None:
        policy_losses = self._extract_losses("policy_stats")

        losses = policy_losses.get("loss", [])
        if not losses:
            self._plot_empty(ax)
            return

        colors = sns.color_palette("colorblind")
        stds = np.zeros_like(losses)  # for backwards compatibility
        self._plot_with_ci(ax, self.epochs, losses, stds, "", colors[0])
        ax.set_title("Policy Loss")
        ax.set_ylabel("Loss")
        # ax.legend()
        self._set_consistent_x_axis(ax)

    def _plot_selfmodel_losses(self, ax: plt.Axes) -> None:
        smodel_losses = self._extract_losses("self_model_stats")

        # look for loss keys that match the pattern <name>_loss where name is a single word
        fast_losses = []
        for key, values in smodel_losses.items():
            if key.endswith("_loss") and values:
                prefix = key.split("_")[0]
                if prefix.isalpha() and prefix.islower():
                    fast_name = prefix + "_loss"
                    fast_losses = values

        if not fast_losses or all(loss == 0 for loss in fast_losses):
            # if losses are all 0, we used zero_<fast_name> so plotting is superfluous
            self._plot_empty(ax)
            return

        colors = sns.color_palette("colorblind")

        stds = np.zeros_like(fast_losses)  # for backwards compatibility
        display_name = fast_name.replace("_", " ")
        self._plot_with_ci(ax, self.epochs, fast_losses, stds, display_name, colors[0])

        ax.set_title("Self Model Loss")
        ax.set_ylabel("Loss")
        ax.legend()
        self._set_consistent_x_axis(ax)

    def _plot_envmodel_losses(self, ax: plt.Axes) -> None:
        model_losses = self._extract_losses("env_model_stats")

        vae_losses = model_losses.get("vae_loss", [])
        mdnrnn_losses = model_losses.get("mdnrnn_loss", [])

        if not vae_losses and not mdnrnn_losses:
            self._plot_empty(ax)
            return

        colors = sns.color_palette("colorblind")

        if vae_losses:
            stds = np.zeros_like(vae_losses)  # for backwards compatibility
            self._plot_with_ci(ax, self.epochs, vae_losses, stds, "vae loss", colors[0])

        if mdnrnn_losses:
            stds = np.zeros_like(mdnrnn_losses)  # for backwards compatibility
            self._plot_with_ci(
                ax, self.epochs, mdnrnn_losses, stds, "mdnrnn loss", colors[1]
            )

        ax.set_title("Environment Model Loss")
        ax.set_ylabel("Loss")
        ax.legend()
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

    def _plot_losses(self, ax: plt.Axes, loss_type: str) -> None:
        losses = self._extract_losses(loss_type)
        if not losses:
            # if the losses are not available, do not include the plot
            self._plot_empty(ax)
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(losses)))
            for (loss_name, loss_values), color in zip(losses.items(), colors):
                loss_name = loss_name.replace("_", " ")
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
            if loss_object is not None:
                loss_dict = loss_object.get_loss_stats_dict()
                for loss, value in loss_dict.items():
                    losses[loss].append(value)
        return losses

    def _has_goal_stats(self) -> bool:
        """Check if goal statistics are available in the epoch stats."""
        if not self.epoch_stats:
            return False

        for stats in self.epoch_stats:
            training_stat = self._get_nested_attr(stats, ["training_stat"])
            self_model = self._get_nested_attr(training_stat, ["self_model_stats"])
            if self_model and hasattr(self_model, "goal_stats"):
                return True
        return False

    def _plot_goal_strategy_stats(self, ax: plt.Axes) -> None:
        """Plot statistics about goal selection strategy if available."""
        goal_stats_over_time = []

        for stats in self.epoch_stats:
            training_stat = self._get_nested_attr(stats, ["training_stat"])
            self_model = self._get_nested_attr(training_stat, ["self_model_stats"])
            if self_model and hasattr(self_model, "goal_stats"):
                goal_stats_over_time.append(self_model.goal_stats)

        if not goal_stats_over_time:
            self._plot_empty(ax)
            return

        steps_per_goal = [stats.avg_steps_per_goal for stats in goal_stats_over_time]
        goal_strategy = goal_stats_over_time[0].goal_strategy

        ax.plot(self.epochs, steps_per_goal, "o-", label="Avg steps per goal")
        ax.set_title(f"Goal Strategy: {goal_strategy}")
        ax.set_ylabel("Average Steps")
        ax.legend()
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
        label = label.replace("_", " ")
        ax.plot(x, y, label=label, color=color, linewidth=2)
        ax.fill_between(
            x,
            np.array(y) - np.array(yerr),
            np.array(y) + np.array(yerr),
            color=color,
            alpha=0.3,
        )

    def _plot_empty(self, ax: plt.Axes) -> None:
        ax.text(0.5, 0.5, "", ha="center", va="center", transform=ax.transAxes)
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

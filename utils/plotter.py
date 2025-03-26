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
        figsize: Tuple[int, int] = (20, 16),
        save_pdf: bool = False,
        pdf_path: Optional[str] = None,
        ncols: Optional[int] = 3,
    ) -> None:
        self._set_plot_style()
        plot_functions = self._get_plot_functions()
        num_plots = len(plot_functions)

        if ncols is None:
            # determine number of columns for a nearly square layout
            ncols = ceil(sqrt(num_plots))
        ncols = max(1, ncols)  # ncols must be >= 1
        nrows = ceil(num_plots / ncols)

        aspect_ratio = nrows / ncols
        fig_width = figsize[0]
        fig_height = figsize[0] * aspect_ratio
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(fig_width, fig_height), squeeze=False
        )

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
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )
        # use a consistent color palette throughout
        self.colors = sns.color_palette("colorblind")
        self.test_color = self.colors[0]  # blue
        self.train_color = self.colors[1]  # orange

    def _get_plot_functions(self) -> Sequence[PlotFunctionTuple]:
        functions = [
            # test and train episodic returns
            (self._plot_episodic_returns, ()),
            # test and train n-step returns
            (self._plot_nstep_returns, ()),
            # intrinsic returns
            (self._plot_returns, ("int_returns",)),
            (self._plot_policy_losses, ()),
            (self._plot_selfmodel_losses, ()),
            (self._plot_envmodel_losses, ()),
            # (self._plot_goal_strategy_stats, ()),
        ]
        return functions

    def _plot_episodic_returns(self, ax: plt.Axes) -> None:
        """Plots the episodic returns with dual y-axes for test and train."""
        ax2 = ax.twinx()

        # test data (left y-axis)
        test_returns, test_returns_std = self._extract_data(
            ["test_collect_stat", "ep_returns_stat"]
        )
        if test_returns:
            self._plot_with_ci(
                ax,
                self.epochs,
                test_returns,
                test_returns_std,
                "test",
                self.test_color,
            )
            # set appropriate y-axis limits
            self._set_y_limits_with_ci(ax, test_returns, test_returns_std)

        # handle train data (right y-axis)
        train_returns, train_returns_std = self._extract_data(
            ["train_collect_stat", "ep_returns_stat"]
        )
        if train_returns:
            self._plot_with_ci(
                ax2,
                self.epochs,
                train_returns,
                train_returns_std,
                "train",
                self.train_color,
            )
            # set appropriate y-axis limits
            self._set_y_limits_with_ci(ax2, train_returns, train_returns_std)

        # display messages if no data
        if not test_returns and not train_returns:
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
        elif not test_returns:
            ax.text(
                0.5,
                0.5,
                "No completed test episodes",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color=self.test_color,
            )
        elif not train_returns:
            ax2.text(
                0.5,
                0.3,
                "No completed train episodes",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color=self.train_color,
            )

        ax.set_title("Mean Episodic Returns")
        ax.set_ylabel("Test Returns")
        ax2.set_ylabel("Train Returns")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper left",
                fontsize=10,
                framealpha=0.9,
            )
        self._set_consistent_x_axis(ax)

    def _plot_nstep_returns(self, ax: plt.Axes) -> None:
        """Plots the n-step returns with dual y-axes for test and train."""
        ax2 = ax.twinx()

        # test data (left y-axis)
        test_returns, test_returns_std = self._extract_data(
            ["test_collect_stat", "returns_stat"]
        )

        self._plot_with_ci(
            ax,
            self.epochs,
            test_returns,
            test_returns_std,
            "test",
            self.test_color,
        )

        # set y-axis limits to include confidence intervals
        self._set_y_limits_with_ci(ax, test_returns, test_returns_std)

        # train data (right y-axis)
        train_returns, train_returns_std = self._extract_data(
            ["train_collect_stat", "returns_stat"]
        )

        self._plot_with_ci(
            ax2,
            self.epochs,
            train_returns,
            train_returns_std,
            "train",
            self.train_color,
        )

        # set y-axis limits to include confidence intervals
        self._set_y_limits_with_ci(ax2, train_returns, train_returns_std)

        ax.set_title("Mean N-Step Returns")
        ax.set_ylabel("Test Returns")
        ax2.set_ylabel("Train Returns")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            fontsize=10,
            framealpha=0.9,
        )
        self._set_consistent_x_axis(ax)

    def _plot_returns(self, ax: plt.Axes, returns_type: str) -> None:
        ax2 = ax.twinx()

        has_test_data = False
        has_train_data = False

        for idx, (collect_type, ax_) in enumerate([("test", ax), ("train", ax2)]):
            color = self.test_color if idx == 0 else self.train_color

            if returns_type == "returns":
                # try episodic returns first
                ep_returns, ep_returns_std = self._extract_data(
                    [f"{collect_type}_collect_stat", f"ep_{returns_type}_stat"]
                )

                if not ep_returns or all(x == 0 for x in ep_returns):
                    # fall back to regular returns if no episodes completed
                    returns, returns_std = self._extract_data(
                        [f"{collect_type}_collect_stat", f"{returns_type}_stat"]
                    )
                else:
                    returns, returns_std = ep_returns, ep_returns_std
            else:
                # for intrinsic rewards, use the regular approach
                returns, returns_std = self._extract_data(
                    [f"{collect_type}_collect_stat", f"{returns_type}_stat"]
                )

            if not returns:
                continue

            if idx == 0:
                has_test_data = True
            else:
                has_train_data = True

            collect_type = collect_type.replace("_", " ")
            self._plot_with_ci(
                ax_,
                self.epochs,
                returns,
                returns_std,
                collect_type,
                color,
            )

            # set appropriate y-axis limits
            self._set_y_limits_with_ci(ax_, returns, returns_std)

        # display message if no data
        if not has_test_data and not has_train_data:
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )

        title_prefix = "Intrinsic" if returns_type == "int_returns" else "Extrinsic"
        ax.set_title(f"Mean {title_prefix} Returns")
        ax.set_ylabel("Test Returns")
        ax2.set_ylabel("Train Returns")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper left",
                fontsize=10,
                framealpha=0.9,
            )
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

        ax.set_title("World Model Loss")
        ax.set_ylabel("Loss")
        ax.legend()
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

        ax.plot(self.epochs, steps_per_goal, "o-", label="Avg. steps per goal")
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

        # plot line with markers at data points
        ax.plot(x, y, label=label, color=color, linewidth=2, marker="o", markersize=4)

        # add confidence interval shading
        ax.fill_between(
            x,
            np.array(y) - np.array(yerr),
            np.array(y) + np.array(yerr),
            color=color,
            alpha=0.3,
        )

    def _set_y_limits_with_ci(
        self,
        ax: plt.Axes,
        values: Sequence[float],
        errors: Sequence[float],
        padding_percent: float = 10.0,
    ) -> None:
        """Sets appropriate y-axis limits to include full confidence intervals."""
        if not values or not errors:
            # set default limits if no data
            ax.set_ylim([0, 1])
            return

        # ensure values and errors have the same length and no None values
        values_clean = []
        errors_clean = []

        for val, err in zip(values, errors):
            if val is not None and err is not None:
                values_clean.append(val)
                errors_clean.append(err)

        # if we don't have clean values, set default limits
        if not values_clean:
            ax.set_ylim([0, 1])
            return

        # calculate the full range including confidence intervals
        values_array = np.array(values_clean)
        errors_array = np.array(errors_clean)

        lower_bounds = values_array - errors_array
        upper_bounds = values_array + errors_array

        # find min and max including confidence intervals
        y_min = np.min(lower_bounds)
        y_max = np.max(upper_bounds)

        # add padding
        y_range = y_max - y_min
        if y_range < 0.01:  # for very small ranges
            y_range = 0.1  # set a minimum range

        padding = y_range * (padding_percent / 100.0)

        if any(y < 0 for y in values_clean):
            ax.set_ylim([y_min - padding, y_max + padding])
        else:
            # no negative values, CIs should not go below zero
            ax.set_ylim([max(0, y_min - padding), y_max + padding])

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
            data = self._get_nested_attr(stats, key_path)
            if data is None:
                # skip None values
                means.append(0.0)
                stds.append(0.0)
            elif isinstance(data, dict):
                means.append(data.get("mean", 0.0))
                stds.append(data.get("std", 0.0))
            elif hasattr(data, "mean") and hasattr(data, "std"):
                # handle None attributes
                mean_val = data.mean if data.mean is not None else 0.0
                std_val = data.std if data.std is not None else 0.0
                means.append(float(mean_val))
                stds.append(float(std_val))
            else:
                # handle other types safely
                try:
                    means.append(float(data))
                except (TypeError, ValueError):
                    means.append(0.0)
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
        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(top=0.95, hspace=0.4, wspace=0.4)
        if save_pdf:
            if pdf_path is None:
                raise ValueError("pdf_path must be provided when save_pdf is True")
            with PdfPages(f"{pdf_path}.pdf") as pdf:
                pdf.savefig(fig, bbox_inches="tight")

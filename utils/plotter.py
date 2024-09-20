import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Dict, Tuple, Any
from tianshou.data import EpochStats


class EpochStatsPlotter:
    def __init__(self, epoch_stats: List[EpochStats]):
        self.epoch_stats = epoch_stats
        self.epochs = [stats.epoch for stats in epoch_stats]

    def plot(self, figsize: Tuple[int, int] = (15, 18), save_pdf: bool = False) -> None:
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
        colours = sns.color_palette("colorblind")

        fig, axs = plt.subplots(3, 2, figsize=figsize)

        self._plot_returns(axs[0, 0], colours)
        self._plot_intrinsic_returns(axs[0, 1], colours)
        self._plot_losses(axs[1, 0], "policy_stats")
        # TODO missing std deviations for intrinsic losses
        self._plot_losses(axs[1, 1], "self_model_stats")
        # TODO intra_episodic plots are wrong, but not because of the plotting code! => test_collect_stat.returns only has one value and train_collect_stat.returns probably only accounts for the external ones
        self._plot_intra_episodic_returns(axs[2, 0], "train")
        self._plot_intra_episodic_returns(axs[2, 1], "test")

        for ax in axs[2, :]:
            ax.set_xlabel("Epoch")

        for ax in axs[:-1, :].flatten():
            ax.tick_params(labelbottom=False)

        plt.tight_layout()
        fig.subplots_adjust(top=1, hspace=0.2, wspace=0.25)

        if save_pdf:
            with PdfPages("statsplot.pdf") as pdf:
                pdf.savefig(fig, bbox_inches="tight")

        plt.show()
        self._print_final_stats()

    def _extract_data(self, key_path: List[str]) -> Tuple[List[float], List[float]]:
        means = []
        stds = []
        for stats in self.epoch_stats:
            data = self._get_nested_attr(stats, key_path)
            if isinstance(data, dict):
                means.append(data.get("mean"))
                stds.append(data.get("std"))
            elif hasattr(data, "mean") and hasattr(data, "std"):
                means.append(data.mean)
                stds.append(data.std)
            else:
                means.append(data)
                stds.append(0)  # Assume no std if not provided
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

    def _extract_losses(
        self, loss_type: str
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        losses = {}
        for stats in self.epoch_stats:
            training_stat = self._get_nested_attr(stats, ["training_stat"])
            if not training_stat:
                continue

            if loss_type == "policy_stats":
                loss_object = self._get_nested_attr(training_stat, ["policy_stats"])
                if loss_object:
                    for loss_name in ["loss", "clip_loss", "vf_loss", "ent_loss"]:
                        if loss_name not in losses:
                            losses[loss_name] = ([], [])
                        loss_stat = getattr(loss_object, loss_name)
                        losses[loss_name][0].append(loss_stat.mean)
                        losses[loss_name][1].append(loss_stat.std)
            elif loss_type == "self_model_stats":
                loss_object = self._get_nested_attr(training_stat, ["self_model_stats"])
                if loss_object:
                    for loss_name in [
                        "icm_loss",
                        "icm_forward_loss",
                        "icm_inverse_loss",
                    ]:
                        if loss_name not in losses:
                            losses[loss_name] = ([], [])
                        loss_value = getattr(loss_object, loss_name)
                        losses[loss_name][0].append(loss_value)
                        losses[loss_name][1].append(
                            0
                        )  # No std available for these losses

        return losses

    def _plot_with_ci(self, ax, x, y, yerr, label, color):
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

    def _plot_returns(self, ax, colours):
        ax2 = ax.twinx()
        test_returns, test_returns_std = self._extract_data(
            ["test_collect_stat", "returns_stat"]
        )
        train_returns, train_returns_std = self._extract_data(
            ["train_collect_stat", "returns_stat"]
        )

        self._plot_with_ci(
            ax, self.epochs, test_returns, test_returns_std, "Test Returns", colours[0]
        )
        self._plot_with_ci(
            ax2,
            self.epochs,
            train_returns,
            train_returns_std,
            "Train Returns",
            colours[1],
        )

        ax.set_title("Returns over Epochs")
        ax.set_ylabel("Test Returns")
        ax2.set_ylabel("Train Returns")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        self._set_consistent_x_axis(ax)

    def _plot_intrinsic_returns(self, ax, colours):
        ax2 = ax.twinx()
        test_returns, test_returns_std = self._extract_data(
            ["test_collect_stat", "int_returns_stat"]
        )
        train_returns, train_returns_std = self._extract_data(
            ["train_collect_stat", "int_returns_stat"]
        )

        self._plot_with_ci(
            ax,
            self.epochs,
            test_returns,
            test_returns_std,
            "Intrinsic Test Returns",
            colours[0],
        )
        self._plot_with_ci(
            ax2,
            self.epochs,
            train_returns,
            train_returns_std,
            "Intrinsic Train Returns",
            colours[1],
        )

        ax.set_title("Intrinsic Returns over Epochs")
        ax.set_ylabel("Test Returns")
        ax2.set_ylabel("Train Returns")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        self._set_consistent_x_axis(ax)

    def _plot_losses(self, ax, loss_type: str):
        losses = self._extract_losses(loss_type)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(losses)))
        for (loss_name, (means, stds)), color in zip(losses.items(), colors):
            self._plot_with_ci(ax, self.epochs, means, stds, loss_name, color)
        ax.set_title(f"{loss_type.replace('_', ' ').title()} over Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        self._set_consistent_x_axis(ax)

    def _plot_intra_episodic_returns(self, ax, data_type: str):
        intra_returns = [
            self._get_nested_attr(stats, [f"{data_type}_collect_stat", "returns"])
            for stats in self.epoch_stats
        ]
        box_positions = np.arange(len(self.epochs))
        ax.boxplot(intra_returns, positions=box_positions)
        ax.set_title(f"Intra-episodic Returns ({data_type.capitalize()})")
        ax.set_ylabel("Returns")
        self._set_consistent_x_axis(ax)

    def _print_final_stats(self):
        final_stats = self.epoch_stats[-1]
        print("Final Key Performance Indicators:")
        print(
            f"Best Reward: {self._get_nested_attr(final_stats, ['info_stat', 'best_reward']):.2f}"
        )
        print(
            f"Total Train Steps: {self._get_nested_attr(final_stats, ['info_stat', 'train_step'])}"
        )
        print(
            f"Total Test Steps: {self._get_nested_attr(final_stats, ['info_stat', 'test_step'])}"
        )
        print(
            f"Final Update Speed: {self._get_nested_attr(final_stats, ['info_stat', 'timing', 'update_speed']):.2f} steps/s"
        )

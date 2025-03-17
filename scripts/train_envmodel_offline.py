import os
import pickle
import logging
import time
import json
from typing import Dict, Any, Literal, Tuple
import matplotlib.pyplot as plt
import numpy as np

import torch
import gymnasium as gym
from datetime import datetime
from models.env_model import EnvModel
from config.config import ConfigManager
from utils.experiment_factory import ExperimentFactory
from utils.experiment_runner import BUFFER_DIR, WEIGHTS_DIR
from core.types import GoalReplayBufferProtocol

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"envmodel_training_{datetime.now().strftime('%d%m%Y-%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger("envmodel_trainer")


def train_envmodel(
    env_name: Literal[
        "frozenlake", "nethack_full", "nethack_score", "nethack_gold"
    ] = "frozenlake",
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 10,
    test_split: float = 0.2,
    patience: int = 5,
    seq_length: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the environment model using offline data."""
    start_time = time.time()
    logger.info(
        f"Starting environment model training with params: env={env_name}, batch_size={batch_size}, "
        f"lr={learning_rate}, epochs={max_epochs}, patience={patience}, seq_length={seq_length}, device={device}"
    )

    # setup
    base_conf_path = "config/offline.yaml"
    policy_config = "random"

    # Set environment-specific configurations
    if env_name == "frozenlake":
        obsnet_config = "discrete"
    else:  # nethack variants
        obsnet_config = "nethack"

    intrinsic_fast_config = "zero_icm"
    intrinsic_slow_config = "zero_her"
    device = torch.device(device)

    config_manager = ConfigManager(base_conf_path)
    subconfigs = {
        "environment": env_name,
        "policy": policy_config,
        "obsnet": obsnet_config,
        "intrinsic/fast": intrinsic_fast_config,
        "intrinsic/slow": intrinsic_slow_config,
    }
    config_manager.create_config(subconfigs)
    logger.info(f"Created configuration with environment: {env_name}")

    factory = ExperimentFactory(config=config_manager)

    full_env_name = config_manager.get("environment.base.name")
    env_config = config_manager.get_except("environment.base", "name")
    env = factory.wrap_env(gym.make(id=full_env_name, **env_config))
    logger.info(f"Created environment: {full_env_name}")

    vae, mdnrnn, _ = factory.create_vae_mdnrnn(env.observation_space, device)
    vae_trainer, mdnrnn_trainer = factory.create_envmodel_trainers(
        vae,
        mdnrnn,
        batch_size,
        learning_rate,
        device,
        use_finetuning=False,
        freeze_envmodel=False,
    )
    env_model = EnvModel(vae, mdnrnn, vae_trainer, mdnrnn_trainer, device=device)
    logger.info(f"Created environment model (VAE + MDNRNN) on device: {device}")

    # load buffer
    buffer_base_path = os.path.join(
        BUFFER_DIR,
        full_env_name.lower(),
        policy_config,
        obsnet_config,
        intrinsic_fast_config,
        intrinsic_slow_config,
    )
    buffer_path = find_most_recent_buffer(buffer_dir=buffer_base_path)
    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)
    logger.info(f"Loaded buffer with {len(buffer)} transitions from {buffer_path}")

    train_buffer, test_buffer, train_starts, test_starts = split_buffer(
        buffer, batch_size, seq_length, test_split
    )
    logger.info(
        f"Split data into {len(train_buffer)} training and {len(test_buffer)} testing transitions "
        f"using {len(train_starts)} training sequences and {len(test_starts)} testing sequences "
        f"of length {seq_length}"
    )

    save_dir = os.path.join(
        WEIGHTS_DIR,
        full_env_name.lower(),
        policy_config,
        obsnet_config,
        intrinsic_fast_config,
        intrinsic_slow_config,
    )
    os.makedirs(save_dir, exist_ok=True)

    logger.info(
        f"Starting training for {max_epochs} epochs with early stopping: patience={patience}"
    )
    best_test_loss = float("inf")
    train_history = []
    test_history = []

    # early stopping variables
    patience_counter = 0
    best_epoch = 0
    best_vae_state = None
    best_mdnrnn_state = None
    initial_vae_state = vae.state_dict().copy()
    initial_mdnrnn_state = mdnrnn.state_dict().copy()
    for epoch in range(max_epochs):
        epoch_start = time.time()

        train_stats = env_model.learn(train_buffer)
        train_vae_loss = train_stats.vae_loss.mean
        train_mdnrnn_loss = train_stats.mdnrnn_loss.mean
        train_mdnrnn_gmm_loss = train_stats.mdnrnn_gmm_loss.mean
        train_mdnrnn_bce_loss = train_stats.mdnrnn_bce_loss.mean
        train_mdnrnn_mse_loss = train_stats.mdnrnn_mse_loss.mean
        train_mean_loss = (train_vae_loss + train_mdnrnn_loss) / 2

        test_stats = env_model.evaluate(test_buffer)
        test_vae_loss = test_stats.vae_loss.mean
        test_mdnrnn_loss = test_stats.mdnrnn_loss.mean
        test_mdnrnn_gmm_loss = test_stats.mdnrnn_gmm_loss.mean
        test_mdnrnn_bce_loss = test_stats.mdnrnn_bce_loss.mean
        test_mdnrnn_mse_loss = test_stats.mdnrnn_mse_loss.mean
        test_mean_loss = (test_vae_loss + test_mdnrnn_loss) / 2

        train_history.append(
            {
                "epoch": epoch + 1,
                "vae_loss": train_vae_loss,
                "mdnrnn_loss": train_mdnrnn_loss,
                "mdnrnn_gmm_loss": train_mdnrnn_gmm_loss,
                "mdnrnn_bce_loss": train_mdnrnn_bce_loss,
                "mdnrnn_mse_loss": train_mdnrnn_mse_loss,
                "mean_loss": train_mean_loss,
            }
        )
        test_history.append(
            {
                "epoch": epoch + 1,
                "vae_loss": test_vae_loss,
                "mdnrnn_loss": test_mdnrnn_loss,
                "mdnrnn_gmm_loss": test_mdnrnn_gmm_loss,
                "mdnrnn_bce_loss": test_mdnrnn_bce_loss,
                "mdnrnn_mse_loss": test_mdnrnn_mse_loss,
                "mean_loss": test_mean_loss,
            }
        )

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{max_epochs} ({epoch_time:.2f}s) | "
            f"Train: VAE={train_vae_loss:.4f}, MDNRNN={train_mdnrnn_loss:.4f} "
            f"(GMM={train_mdnrnn_gmm_loss:.4f}, BCE={train_mdnrnn_bce_loss:.4f}, MSE={train_mdnrnn_mse_loss:.4f}) "
            f"Mean={train_mean_loss:.4f} | "
            f"Test: VAE={test_vae_loss:.4f}, MDNRNN={test_mdnrnn_loss:.4f} "
            f"(GMM={test_mdnrnn_gmm_loss:.4f}, BCE={test_mdnrnn_bce_loss:.4f}, MSE={test_mdnrnn_mse_loss:.4f}) "
            f"Mean={test_mean_loss:.4f}"
        )

        # track best model
        if test_mean_loss < best_test_loss:
            best_test_loss = test_mean_loss
            best_epoch = epoch + 1
            patience_counter = 0

            # store the best model states
            best_vae_state = vae.state_dict().copy()
            best_mdnrnn_state = mdnrnn.state_dict().copy()

            logger.info(f"New best model with test loss {test_mean_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement for {patience_counter} epochs (best: {best_test_loss:.4f} at epoch {best_epoch})"
            )

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # save only the best model at the end of training
    if best_epoch > 0:
        logger.info(
            f"Saving best model from epoch {best_epoch} with test loss {best_test_loss:.4f}"
        )
    else:
        logger.warning("No improvement was found during training, saving initial model")

    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    vae_path = os.path.join(save_dir, f"{timestamp}_vae.pth")
    mdnrnn_path = os.path.join(save_dir, f"{timestamp}_mdnrnn.pth")

    if best_vae_state is None or best_mdnrnn_state is None:
        best_vae_state = initial_vae_state
        best_mdnrnn_state = initial_mdnrnn_state

    torch.save(best_vae_state, vae_path)
    torch.save(best_mdnrnn_state, mdnrnn_path)
    logger.info(f"Model saved to {vae_path} and {mdnrnn_path}")

    # save training history
    history = {
        "train": train_history,
        "test": test_history,
        "config": {
            "env_name": env_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "test_split": test_split,
            "patience": patience,
            "seq_length": seq_length,
            "device": str(device),
            "full_env_name": full_env_name,
            "buffer_path": buffer_path,
            "num_train_sequences": len(train_starts),
            "num_test_sequences": len(test_starts),
            "best_epoch": best_epoch,
        },
    }
    history_path = os.path.join(save_dir, f"training_history_{timestamp}.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    plot_path = os.path.join(save_dir, f"training_plot_{timestamp}.png")
    plot_training_history(history, plot_path)
    logger.info(f"Training plot saved to {plot_path}")

    summary_path = save_model_summary(history, save_dir, timestamp)
    logger.info(f"Model summary saved to {summary_path}")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    logger.info(f"Best test loss: {best_test_loss:.4f}")
    logger.info(f"Training history saved to {history_path}")

    return best_test_loss, best_epoch


def find_most_recent_buffer(buffer_dir: str) -> str:
    dt = "%d%m%Y-%H%M%S"
    valid_buffers = []
    for f in os.listdir(buffer_dir):
        if f.endswith("_buffer.pkl"):
            try:
                timestamp = datetime.strptime(f.split("_")[0], dt)
                valid_buffers.append((f, timestamp))
            except ValueError:
                continue

    if not valid_buffers:
        raise ValueError(f"No valid buffer files found in {buffer_dir}")

    latest_buffer = max(valid_buffers, key=lambda x: x[1])[0]
    return os.path.join(buffer_dir, latest_buffer)


def split_buffer(
    buffer: GoalReplayBufferProtocol,
    batch_size: int,
    seq_length: int,
    test_split: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the buffer into train and test sets, ensuring enough valid sequences for training by maintaining temporal coherence within sequences."""
    total_indices = buffer.sample_indices(0)
    total_size = len(total_indices)
    logger.info(f"Total transitions in buffer: {total_size}")

    # this should be long enough to capture temporal dependencies
    logger.info(f"Using sequence length of {seq_length} for RNN training")

    # find episode boundaries to avoid crossing them
    done_indices = np.where(buffer.done[total_indices])[0]
    episode_starts = np.concatenate(([0], done_indices + 1))
    episode_ends = np.concatenate((done_indices, [total_size - 1]))

    # try to find valid sequences, reducing sequence length if necessary
    original_seq_length = seq_length
    valid_starts = []
    while seq_length >= 2:
        valid_starts = []
        for start, end in zip(episode_starts, episode_ends):
            # only use episodes that are long enough
            if end - start + 1 >= seq_length:
                # add all possible starting points within this episode
                valid_starts.extend(range(start, end - seq_length + 2))

        # check if we have enough valid sequences
        min_required = max(
            2, int(batch_size * 1.5)
        )  # need at least batch_size * 1.5 sequences
        if len(valid_starts) >= min_required:
            break

        # reduce sequence length and try again
        seq_length = max(2, seq_length - 2)

    if len(valid_starts) < 2:
        raise ValueError(
            f"Not enough valid sequences found in buffer even after reducing sequence length to {seq_length}. "
            f"Found {len(valid_starts)} sequences. The buffer may not contain enough transitions."
        )

    if seq_length != original_seq_length:
        logger.warning(
            f"Reduced sequence length from {original_seq_length} to {seq_length} "
            f"to ensure enough valid sequences for training"
        )

    logger.info(f"Found {len(valid_starts)} valid sequences of length {seq_length}")

    # shuffle the valid starting points to break temporal order between sequences
    # while maintaining temporal coherence within sequences
    np.random.shuffle(valid_starts)

    # split into train and test sets
    split_idx = int(len(valid_starts) * (1 - test_split))
    train_starts = valid_starts[:split_idx]
    test_starts = valid_starts[split_idx:]

    # create sequence indices for training and testing
    train_idxs = []
    for start in train_starts:
        train_idxs.extend(total_indices[start : start + seq_length])

    test_idxs = []
    for start in test_starts:
        test_idxs.extend(total_indices[start : start + seq_length])

    train_idxs = np.array(train_idxs)
    test_idxs = np.array(test_idxs)

    train_buffer = buffer[train_idxs]
    test_buffer = buffer[test_idxs]
    return train_buffer, test_buffer, train_starts, test_starts


def plot_training_history(history: Dict[str, Any], save_path: str) -> None:
    """Plot training and test losses over epochs."""
    train_data = history["train"]
    test_data = history["test"]

    epochs = [entry["epoch"] for entry in train_data]

    _, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

    # VAE loss
    axes[0].plot(
        epochs, [entry["vae_loss"] for entry in train_data], "b-", label="Train"
    )
    axes[0].plot(epochs, [entry["vae_loss"] for entry in test_data], "r-", label="Test")
    axes[0].set_ylabel("VAE Loss")
    axes[0].set_title("Training Progress")
    axes[0].legend()
    axes[0].grid(True)

    # GMM loss
    axes[1].plot(
        epochs,
        [entry["mdnrnn_gmm_loss"] for entry in train_data],
        "b-",
        label="Train",
    )
    axes[1].plot(
        epochs,
        [entry["mdnrnn_gmm_loss"] for entry in test_data],
        "r-",
        label="Test",
    )
    axes[1].set_ylabel("GMM Loss")
    axes[1].legend()
    axes[1].grid(True)

    # BCE loss
    axes[2].plot(
        epochs,
        [entry["mdnrnn_bce_loss"] for entry in train_data],
        "b-",
        label="Train",
    )
    axes[2].plot(
        epochs,
        [entry["mdnrnn_bce_loss"] for entry in test_data],
        "r-",
        label="Test",
    )
    axes[2].set_ylabel("BCE Loss")
    axes[2].legend()
    axes[2].grid(True)

    # MSE loss
    axes[3].plot(
        epochs,
        [entry["mdnrnn_mse_loss"] for entry in train_data],
        "b-",
        label="Train",
    )
    axes[3].plot(
        epochs,
        [entry["mdnrnn_mse_loss"] for entry in test_data],
        "r-",
        label="Test",
    )
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("MSE Loss")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_model_summary(history: Dict[str, Any], save_dir: str, timestamp: str) -> str:
    """Save a summary of the model's performance."""
    train_data = history["train"]
    test_data = history["test"]
    config = history["config"]

    best_test_epoch = min(test_data, key=lambda x: x["mean_loss"])
    best_train_epoch = min(train_data, key=lambda x: x["mean_loss"])

    summary = {
        "training_config": config,
        "best_test_performance": {
            "epoch": best_test_epoch["epoch"],
            "mean_loss": best_test_epoch["mean_loss"],
            "vae_loss": best_test_epoch["vae_loss"],
            "mdnrnn_loss": best_test_epoch["mdnrnn_loss"],
        },
        "best_train_performance": {
            "epoch": best_train_epoch["epoch"],
            "mean_loss": best_train_epoch["mean_loss"],
            "vae_loss": best_train_epoch["vae_loss"],
            "mdnrnn_loss": best_train_epoch["mdnrnn_loss"],
        },
        "final_performance": {"train": train_data[-1], "test": test_data[-1]},
        "total_epochs": len(train_data),
        "early_stopped": len(train_data) < config["max_epochs"],
        "timestamp": timestamp,
    }

    summary_path = os.path.join(save_dir, f"model_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path


if __name__ == "__main__":
    # using hardcoded parameters makes for simpler debugging
    # options: "frozenlake", "nethack_full", "nethack_score", "nethack_gold"
    env_name = "frozenlake"

    best_loss, best_epoch = train_envmodel(
        env_name=env_name,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=10,
        test_split=0.2,
        patience=5,
        seq_length=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

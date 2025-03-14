import os
import pickle
import numpy as np
import logging
import time
import json
from typing import Dict, Any
import matplotlib.pyplot as plt

import torch
import gymnasium as gym
from datetime import datetime
from models.env_model import EnvModel
from config.config import ConfigManager
from utils.experiment_factory import ExperimentFactory
from utils.experiment_runner import BUFFER_DIR, WEIGHTS_DIR

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
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 10,
    test_split: float = 0.2,
    patience: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the environment model using offline data."""
    start_time = time.time()
    logger.info(
        f"Starting environment model training with params: batch_size={batch_size}, "
        f"lr={learning_rate}, epochs={max_epochs}, patience={patience}, device={device}"
    )

    # setup
    base_conf_path = "config/vanilla_offline.yaml"
    env_name = "frozenlake"
    policy_config = "random"
    obsnet_config = "discrete"
    intrinsic_fast_config = "zero_icm"
    intrinsic_slow_config = "zero_her"
    is_goal_aware = True
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
        "goal" if is_goal_aware else "vanilla",
    )
    buffer_path = find_most_recent_buffer(buffer_dir=buffer_base_path)
    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)
    logger.info(f"Loaded buffer with {len(buffer)} transitions from {buffer_path}")

    # temporal order is important for RNN training
    total_indices = buffer.sample_indices(0)
    split_idx = int(len(total_indices) * (1 - test_split))
    train_idxs = total_indices[:split_idx]
    test_idxs = total_indices[split_idx:]
    train_buffer = buffer[train_idxs]
    test_buffer = buffer[test_idxs]
    logger.info(
        f"Split data into {len(train_buffer)} training and {len(test_buffer)} testing transitions"
    )

    save_dir = os.path.join(
        WEIGHTS_DIR,
        full_env_name.lower(),
        policy_config,
        obsnet_config,
        intrinsic_fast_config,
        intrinsic_slow_config,
        "goal" if is_goal_aware else "vanilla",
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

        # save if test loss improved
        if test_mean_loss < best_test_loss:
            best_test_loss = test_mean_loss
            best_epoch = epoch + 1
            patience_counter = 0

            vae_path = os.path.join(save_dir, "vae.pth")
            mdnrnn_path = os.path.join(save_dir, "mdnrnn.pth")

            torch.save(vae.state_dict(), vae_path)
            torch.save(mdnrnn.state_dict(), mdnrnn_path)
            logger.info(f"Saved best model with test loss {test_mean_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement for {patience_counter} epochs (best: {best_test_loss:.4f} at epoch {best_epoch})"
            )

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # save training history
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    history = {
        "train": train_history,
        "test": test_history,
        "config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "test_split": test_split,
            "patience": patience,
            "device": str(device),
            "env_name": full_env_name,
            "buffer_path": buffer_path,
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


def save_model_summary(history: Dict[str, Any], save_dir: str, timestamp: str) -> None:
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
    train_envmodel(
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=10,
        test_split=0.2,
        patience=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

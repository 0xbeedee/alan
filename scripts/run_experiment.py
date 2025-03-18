import argparse
import torch

from utils.experiment_runner import ExperimentRunner


def main(
    base_config_path,
    env_config,
    policy_config,
    obsnet_config,
    intrinsic_fast_config,
    intrinsic_slow_config,
    model_config,
    goal_strategy_config,
    device,
):
    runner = ExperimentRunner(
        base_config_path,
        env_config,
        policy_config,
        obsnet_config,
        intrinsic_fast_config,
        intrinsic_slow_config,
        model_config,
        goal_strategy_config,
        device,
    )

    runner.setup()
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            The Reinforcement Learning Experiment Runner.

            This script sets up and runs reinforcement learning experiments with customizable environments, policies, and observation networks.

            Experiments are defined through YAML configuration files (usually located in the config/ directory), allowing for easy parameter tuning and reproducibility.
            """
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the base configuration file (including .yaml extension)",
        metavar="BASE_CONFIG_PATH",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        required=True,
        help="Name of the environment config file (without .yaml extension)",
        metavar="ENV_CONFIG",
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        required=True,
        help="Name of the policy config file (without .yaml extension)",
        metavar="POLICY_CONFIG",
    )
    parser.add_argument(
        "-o",
        "--obsnet",
        type=str,
        required=True,
        help="Name of the observation network config file (without .yaml extension)",
        metavar="OBSNET_CONFIG",
    )
    parser.add_argument(
        "-if",
        "--intrinsic_fast",
        type=str,
        required=True,
        help="Name of the fast intrinsic module config file (without .yaml extension)",
        metavar="INTRINSIC_CONFIG",
    )
    parser.add_argument(
        "-is",
        "--intrinsic_slow",
        type=str,
        required=True,
        help="Name of the slow intrinsic module config file (without .yaml extension)",
        metavar="INTRINSIC_CONFIG",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Name of the config file for the models (without .yaml extension)",
        metavar="MODEL_CONFIG",
    )
    parser.add_argument(
        "-g",
        "--goal_strategy",
        type=str,
        default="zero",
        help="Name of the goal selection strategy config file (without .yaml extension)",
        metavar="GOAL_STRATEGY_CONFIG",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
        metavar="DEVICE",
    )

    args = parser.parse_args()
    main(
        args.config,
        args.env,
        args.policy,
        args.obsnet,
        args.intrinsic_fast,
        args.intrinsic_slow,
        args.model,
        args.goal_strategy,
        torch.device(args.device),
    )

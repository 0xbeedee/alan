import argparse
import torch

from utils.experiment_runner import ExperimentRunner


def main(
    base_config_path,
    env_config,
    policy_config,
    obsnet_config,
    intrinsic_config,
    model_config,
    use_kb,
    save_kb,
    device,
):
    runner = ExperimentRunner(
        base_config_path,
        env_config,
        policy_config,
        obsnet_config,
        intrinsic_config,
        model_config,
        use_kb,
        save_kb,
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
        "-i",
        "--intrinsic",
        type=str,
        required=True,
        help="Name of the intrinsic module config file (without .yaml extension)",
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
        "-k",
        "--use_kb",
        action="store_true",
        help="If set, use the knowledge base (and the associated machinery)",
    )
    parser.add_argument(
        "-s",
        "--save_kb",
        action="store_true",
        help="If set, save the knowledge base, persisting it for future runs",
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
        args.intrinsic,
        args.model,
        args.use_kb,
        args.save_kb,
        torch.device(args.device),
    )

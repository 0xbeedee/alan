import argparse
import torch
import multiprocessing

from utils.experiment_runner import ExperimentRunner


def main(
    base_config_path,
    env_config,
    policy_config,
    obsnet_config,
    intrinsic_fast_config,
    intrinsic_slow_config,
    envmodel_config,
    selfmodel_config,
    device,
    use_kb,
    save_kb,
    enable_dream,
):
    runner = ExperimentRunner(
        base_config_path,
        env_config,
        policy_config,
        obsnet_config,
        intrinsic_fast_config,
        intrinsic_slow_config,
        envmodel_config,
        selfmodel_config,
        device,
        use_kb=use_kb,
        save_kb=save_kb,
        enable_dream=enable_dream,
    )

    runner.setup()
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the base config file",
        metavar="CONFIG_PATH",
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
        metavar="INTRINSIC_FAST_CONFIG",
    )
    parser.add_argument(
        "-is",
        "--intrinsic_slow",
        type=str,
        required=True,
        help="Name of the slow intrinsic module config file (without .yaml extension)",
        metavar="INTRINSIC_SLOW_CONFIG",
    )
    parser.add_argument(
        "-em",
        "--envmodel",
        type=str,
        required=True,
        help="Name of the environment model config file (without .yaml extension)",
        metavar="ENVMODEL_CONFIG",
    )
    parser.add_argument(
        "-sm",
        "--selfmodel",
        type=str,
        required=True,
        help="Name of the self model config file (without .yaml extension)",
        metavar="SELFMODEL_CONFIG",
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
    parser.add_argument(
        "--use-kb",
        action="store_true",
        help="Use knowledge base",
    )
    parser.add_argument(
        "--save-kb",
        action="store_true",
        help="Save knowledge base after training",
    )
    parser.add_argument(
        "--enable-dream",
        action="store_true",
        help="Enable dreaming",
    )

    args = parser.parse_args()
    main(
        args.config,
        args.env,
        args.policy,
        args.obsnet,
        args.intrinsic_fast,
        args.intrinsic_slow,
        args.envmodel,
        args.selfmodel,
        torch.device(args.device),
        args.use_kb,
        args.save_kb,
        args.enable_dream,
    )

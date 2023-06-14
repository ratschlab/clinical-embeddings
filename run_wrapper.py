#!/usr/bin/env python3

# ===========================================================
#
# Run a hyperparameter search
#
# ===========================================================
import argparse
import datetime
import logging
import os
import random
import shlex
import shutil
import subprocess
import time
from itertools import product
from pathlib import Path

import coloredlogs
import yaml
from coolname import generate_slug
from tqdm import tqdm

from icu_benchmarks.models.utils import get_modalities_list

# TODO: remember already launched configs, enable continuation of random search
# TODO: gather results and hyperparams of finished runs

# ========================
# GLOBAL
# ========================
LOGGING_LEVEL = "INFO"
EXPERIMENT_GLOBALS = {"base_gin", "accelerate_config", "task", "seeds"}

# ========================
# Argparse
# ========================
def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    # ------------------
    # General options
    # ------------------
    parser.add_argument(
        "-n", "--name", type=str, default="test", help="Experiment Name"
    )
    parser.add_argument("-d", "--directory", type=str, default="./logs")
    parser.add_argument("--clear_directory", default=False, action="store_true")
    parser.add_argument(
        "--config", required=True, type=str, help="Configuration yaml file"
    )
    parser.add_argument("--print", default=False, action="store_true")
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="Seconds to wait between submissions"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=-1,
        help="Number of runs to perform; default=-1 run all configs",
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="Shuffle configs before running; e.g. random search",
    )

    parser_compute = parser.add_argument_group("Compute")
    parser_compute.add_argument("-g", "--gpus", type=int, default=0)
    parser_compute.add_argument("--gpu_type", type=str, default=None)
    parser_compute.add_argument("-c", "--cores", type=int, default=2)
    parser_compute.add_argument(
        "-m", "--memory", type=int, default=4, help="Requests {m}G of memory per core"
    )
    parser_compute.add_argument(
        "-t", "--time", type=int, default=1, help="Request {t} hours of job duration"
    )
    parser_compute.add_argument(
        "-e",
        "--exclude",
        default=[],
        nargs="+",
        help="Nodes to exclude from submission",
    )

    args = parser.parse_args()
    return args


def create_slurm_command(args: argparse.Namespace, compute_config: dict = {}) -> str:

    args_dict = vars(args)
    for key in ["gpus", "gpu_type", "cores", "memory", "time"]:
        if key not in compute_config:
            compute_config[key] = args_dict[key]

    cmd = "sbatch"
    cmd += f" --time {compute_config['time']}:00:00"
    cmd += f" -c {compute_config['cores']}"
    cmd += f" --mem-per-cpu {compute_config['memory']}000"

    if len(args.exclude) > 0:
        nodelist = ",".join(args.exclude)
        logging.info(f"[COMPUTE] excluding nodes: {nodelist}")
        cmd += f" --exclude={nodelist}"

    gpu_type = (
        "" if compute_config["gpus"] is None else f":{compute_config['gpu_type']}"
    )
    gpu_cmd = ""
    multi_gpu = compute_config["gpus"] > 1
    if compute_config["gpus"] > 0:
        gpu_cmd += " -p gpu"
        gpu_cmd += f" --gres gpu{gpu_type}:{compute_config['gpus']}"
        cmd += gpu_cmd

    cmd += f" --job-name={args.name}"

    log_dir = os.path.join(args.directory, args.name, "slurm")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cmd += f" --output={log_dir}/slurm-%j.out"

    logging.info(f"[COMPUTE] name:  {args.name}")
    logging.info(f"[COMPUTE] time:  {compute_config['time']}:00:00")
    logging.info(f"[COMPUTE] cores: {compute_config['cores']}")
    logging.info(f"[COMPUTE] mem:   {compute_config['memory']}000")
    logging.info(f"[COMPUTE] gpu:   {gpu_cmd}")

    return cmd, multi_gpu


def create_configurations(config_path: str, args: argparse.Namespace) -> list:

    with open(config_path, "r") as stream:
        try:
            hp_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)

    base_config = {}
    params = hp_config["params"]
    search_params = []

    for k, v in params.items():
        if type(v) == list:
            search_params.append(k)
        else:
            base_config[k] = v

    # global experiment settings
    for k in EXPERIMENT_GLOBALS:
        base_config[k] = hp_config[k]

    logging.info(f"[CONFIG] {len(search_params)} search parameters")
    logging.info(f"[CONFIG] {search_params}")

    configurations = [base_config]
    for p in search_params:
        new_configurations = []
        for v, c in product(params[p], configurations):
            new_c = c.copy()
            if type(v) == list:
                new_c[p] = " ".join(map(str, v))
            else:
                new_c[p] = v
            new_configurations.append(new_c)

        configurations = new_configurations

    return configurations, hp_config


def create_gin(config: dict, args: argparse.Namespace, run_directory: str) -> str:

    # read base gin config
    base_config_f = open(config["base_gin"], "r")
    base_config = base_config_f.read()
    base_config_f.close()

    # create new config by adding search parameters
    final_config_path = os.path.join(run_directory, "launch_config.gin")
    with open(final_config_path, "w") as f:

        # write base config
        f.write(base_config)
        f.write("\n\n")
        f.write("#" + 50 * "=" + "\n")
        f.write("# SWEEP CONFIG\n")
        f.write("#" + 50 * "=" + "\n")

        # add options
        for h_key, h_value in sorted(config.items()):

            if h_key in EXPERIMENT_GLOBALS:
                continue

            if h_key in {"modality_splitting", "mod_split", "mod_split_clustered"}:
                h_value = get_modalities_list(h_value)
                split_class = (
                    "ClusteredSplittedEmbedding"
                    if "cluster" in h_key
                    else "Splitted_Embedding"
                )
                h_key = f"{split_class}.reindex_modalities"

            elif h_key == "DLWrapper.clustering_tensorboard" and h_value:
                h_value = run_directory

            if isinstance(h_value, str) and h_value[0] != "@":
                f.write(f"{h_key} = '{h_value}'\n")
            else:
                f.write(f"{h_key} = {h_value}\n")

        # set task
        f.write(f"ICUVariableLengthLoaderTables.task = '{config['task']}'\n")

    return final_config_path


def create_run_dir_and_gin(config: dict, args: argparse.Namespace) -> str:

    run_name = generate_slug(3)
    full_run_directory = os.path.join(args.directory, args.name, run_name)

    gin_config_path = ""
    if not args.print:
        os.makedirs(full_run_directory)
        gin_config_path = create_gin(config, args, full_run_directory)

    return full_run_directory, gin_config_path


def config_to_cmd_str(
    config: dict,
    args: argparse.Namespace,
    split_seeds: bool = False,
    multi_gpu_flag: bool = False,
) -> str:

    # seed arguments
    if not split_seeds:
        seed_args = [f"-sd {' '.join(map(str, config['seeds']))}"]
    else:
        seed_args = [f"-sd {s}" for s in config["seeds"]]

    # Create custom GIN
    run_directory, gin_config_path = create_run_dir_and_gin(config, args)

    run_cmds = []
    for seed_arg in seed_args:

        # accelerate base launch command
        run_cmd = ""
        run_cmd += "accelerate launch "
        run_cmd += f"--config_file {config['accelerate_config']} "

        # add master port if using multiple GPUs
        if multi_gpu_flag:
            random_port = random.randint(20000, 50000)
            run_cmd += f"--main_process_port {random_port} "
            # logging.info(f"[DISTRIBUTED] master port: {random_port}")

        # add python run script
        run_cmd += "./icu_benchmarks/run.py train "

        run_cmd += f"-c {gin_config_path} "
        run_cmd += f"-l {run_directory} "
        run_cmd += seed_arg

        run_cmds.append(run_cmd)

    return run_cmds


def run_on_slurm(
    config: dict,
    slurm_command: str,
    args: argparse.Namespace,
    multi_gpu_flag: bool = False,
) -> int:

    config_cmds = config_to_cmd_str(
        config, args, split_seeds=True, multi_gpu_flag=multi_gpu_flag
    )
    for config_cmd in config_cmds:

        run_cmd = ' --wrap="'
        run_cmd += config_cmd
        run_cmd += '"'

        full_cmd = slurm_command + run_cmd
        # logging.info(f"Run: {full_cmd}")

        completed_process = subprocess.run(
            shlex.split(full_cmd), stdout=subprocess.DEVNULL, check=True
        )

    return True


# ========================
# MAIN
# ========================
def main():
    """Training Script procedure"""

    # Parse CMD arguments
    args = parse_arguments()

    logging.info("[CONFIG] Creating configurations")
    configurations, raw_config = create_configurations(args.config, args)
    logging.info(f"[CONFIG] Composed {len(configurations)} configs")

    # Check experiment directory for existing results
    experiment_directory = os.path.join(args.directory, args.name)
    if os.path.isdir(experiment_directory):
        logging.info(
            f"[DIRECTORY] Experiment directory: {experiment_directory} already exists"
        )

        if args.clear_directory:
            clear_directory = True
        else:
            user_input = input(
                "[DIRECTORY] Clear experiment directory ? True / False: "
            )
            clear_directory = user_input.capitalize() == "True"

        if clear_directory:
            logging.warning(f"[DIRECTORY] Clearing existing experiment directory.")
            shutil.rmtree(experiment_directory)

    # Generate slurm submit command
    compute_config = {} if "compute" not in raw_config else raw_config["compute"]
    slurm_cmd, multi_gpu_flag = create_slurm_command(args, compute_config)

    # shuffle list
    if args.shuffle:
        logging.info(f"[CONFIG] shuffle configurations")
        random.shuffle(configurations)

    # Limit configs and shuffle
    if args.num_runs > 0:
        logging.info(f"[CONFIG] cut run dict to max: {args.num_runs}")
        configurations = configurations[: args.num_runs]

    # shuffle to list or dummy print
    logging.info(
        f"[SLURM] Submitting {len(configurations)} runs with name: {args.name}"
    )
    for config in tqdm(configurations):
        if args.print:
            logging.warning(f"[EXEC] printing only")
            logging.info(config_to_cmd_str(config, args))
        else:
            run_on_slurm(config, slurm_cmd, args, multi_gpu_flag=multi_gpu_flag)

        # to avoid race-conditions in file creation
        time.sleep(args.sleep)

    # ------------------------
    # Cleanup
    # ------------------------
    logging.info(40 * "=")
    logging.info("Finished")
    logging.info(40 * "=")


# ========================
# SCRIPT ENTRY
# ========================
if __name__ == "__main__":

    # set logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s | %(message)s"
    )
    coloredlogs.install(level=LOGGING_LEVEL)

    # run script
    main()

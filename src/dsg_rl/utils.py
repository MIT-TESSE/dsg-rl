###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################


import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from ray.rllib.agents import ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import grid_search
from tesse.msgs import Camera
from tesse_gym import ObservationConfig, get_network_config
from tesse_gym.core.utils import set_all_camera_params
from tesse_gym.observers.dsg.scene_graph import get_scene_graph_task
from tesse_gym.tasks.goseek import GoSeek


def populate_rllib_config(
    default_config: Dict[str, Any], user_config: str
) -> Dict[str, Any]:
    """Add or edit items from an rllib config.
    If `user_config` contains a value wrapped in the string
    `grid_search([])`, the value will be given as and
    rllib.tune `grid_search` option.

    Args:
        default_config (Dict[str, Any]): A default rllib config
            (e.g., for ppo).
        user_config (str): Path to configuration file.

    Returns:
        Dict[str, Any]: `default_config` with `user_config` items
            added. If there are matching keys, `user_config` takes
            precedence.
    """
    with open(user_config) as f:
        user_config = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in user_config.items():
        if isinstance(value, str) and "grid_search" in value:
            parsed_value = [
                float(x) for x in value.split("([")[1].split("])")[0].split(",")
            ]
            default_config[key] = grid_search(parsed_value)
        else:
            default_config[key] = user_config[key]

    return default_config


def check_for_tesse_instances() -> None:
    """Raise exception if TESSE instances are already running."""
    if all(
        s in subprocess.run(["ps", "aux"], capture_output=True).stdout.decode("utf-8")
        for s in ["goseek-", ".x86_64"]
    ):
        raise EnvironmentError("TESSE is already running")


class GOSEEKGoalCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        def _get_unwrapped_env_value(f):
            return np.array([[f(env) for env in base_env.get_unwrapped()]]).mean()

        mean_found_targets = _get_unwrapped_env_value(lambda x: x.n_found_targets)
        mean_collisions = _get_unwrapped_env_value(lambda x: x.n_collisions)
        n_visited_nodes = _get_unwrapped_env_value(
            lambda x: x.get_observers()["dsg"].get_visited_nodes()
        )
        episode.custom_metrics["found_targets"] = mean_found_targets
        episode.custom_metrics["collisions"] = mean_collisions
        episode.custom_metrics["visited_nodes"] = n_visited_nodes

        if hasattr(base_env.get_unwrapped()[0], "visited_cells"):
            mean_visited_cells = _get_unwrapped_env_value(
                lambda x: len(x.visited_cells)
            )
            episode.custom_metrics["n_visited_cells"] = mean_visited_cells


def init_function(env: GoSeek) -> None:
    """Initialization function called by the
    TesseyGym environment."""
    cnn_shape = (5, 120, 160)
    set_all_camera_params(
        env, height_in_pixels=cnn_shape[1], width_in_pixels=cnn_shape[2]
    )


def make_goseek_env(config: Dict[str, Any]) -> GoSeek:
    """Generate GOSEEK environment.

    Args:
        config (Dict[str, Any]): GOSEEK configuration.

    Returns:
        GoSeek: TesseGym environment configured for the GOSEEK task.
    """
    cnn_shape = (5, 120, 160)

    observation_config = ObservationConfig(
        modalities=[getattr(Camera, m) for m in config.pop("modalities")],
        height=cnn_shape[1],
        width=cnn_shape[2],
        pose=True,
        use_dict=True,
    )

    worker_index = config.worker_index
    vector_index = config.vector_index
    N_ENVS_PER_WORKER = 1

    rank = worker_index - 1 + (N_ENVS_PER_WORKER + 1) * vector_index
    scenes = config.pop("SCENES")
    scene = scenes[rank % len(scenes)]

    if "RANK" in config.keys():
        config_rank = config.pop("RANK")
        rank = config_rank if isinstance(config_rank, int) else config_rank[rank]

    print(
        f"MAKING GOSEEK ENV w/ rank: {rank}, inds: ({worker_index}, {vector_index}, scene: {scene})"
    )

    env = get_scene_graph_task(
        GoSeek,
        build_path=str(config.pop("FILENAME")),
        network_config=get_network_config(
            simulation_ip="localhost", own_ip="localhost", worker_id=rank
        ),
        observation_config=observation_config,
        init_function=init_function,
        scene_id=scene,
        **config,
    )
    return env


def timesteps_to_train_itrs(batch_size: int, save_freq_timesteps: int) -> int:
    """Get save frequency in training iterations"""
    return save_freq_timesteps // batch_size


def get_video_log_path(log_path: str, experiment_name: str) -> str:
    """Get video log path."""
    if log_path == "None":
        return None
    else:
        return f"{log_path}/{experiment_name}"


def get_ppo_train_config(
    user_config: str, experiment_name: str
) -> Tuple[Dict[str, Any], str]:
    """Get configuration for Rllib's PPO.

    Args:
        user_config (str): Path to DSG-RL configuration file.
        experiment_name (str): Experiment name.

    Returns:
        Tuple[Dict[str, Any], str]:
            - Configuration for PPO
            - log directory
    """
    config = ppo.DEFAULT_CONFIG.copy()
    config["callbacks"] = GOSEEKGoalCallbacks
    config = populate_rllib_config(config, user_config)
    config["env_config"]["video_log_path"] = get_video_log_path(
        config["env_config"]["video_log_path"], experiment_name
    )
    local_dir = config.pop("local_dir")
    save_freq_timesteps = config.pop("ckpt_save_freq_timesteps")
    save_freq = timesteps_to_train_itrs(config["train_batch_size"], save_freq_timesteps)
    config["evaluation_interval"] = save_freq

    return config, local_dir


def get_ppo_eval_config(
    user_config: str,
    ckpt: str,
    log_path: str,
    episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Get configuration for evaluating and Rllib
    PPO model.

    Args:
        user_config (str): Path to DSG-RL configuration file.
        ckpt (str): _description_
        log_path (str): _description_
        episodes (Optional[int], optional): _description_. Defaults to None.

    Returns:
        Dict[str, Any]: Configuration for evaluating and Rllib
            PPO model.
    """
    config = ppo.DEFAULT_CONFIG.copy()
    config["callbacks"] = GOSEEKGoalCallbacks
    config = populate_rllib_config(config, user_config)
    config_path = Path(user_config)

    # set train workers to 0
    config["num_workers"] = 0
    config["num_envs_per_worker"] = 1

    # override number of evaluation episode
    if episodes is not None:
        config["evaluation_num_episodes"] = int(episodes)

    config.pop("ckpt_save_freq_timesteps")  # not used for eval

    # configure video logging
    config_path = Path(user_config)
    n_episodes = config["evaluation_num_episodes"]
    results_path = Path(log_path)
    results_path.mkdir(exist_ok=True, parents=True)
    config["env_config"]["video_log_path"] = get_video_log_path(
        results_path, f"{config_path.stem}_{ckpt.name}_{n_episodes}_episode_videos"
    )

    config.pop("local_dir")  # isn't used by evaluator

    return config


def log_eval_results(
    results: Dict[str, Any],
    config: Dict[str, Any],
    config_path: Path,
    ckpt: Path,
    log_path: str,
) -> None:
    """Log evaluation results file file.

    Args:
        results (Dict[str, Any]): Dictionary of metric-value pairs.
        config (Dict[str, Any]): Configuration used for evaluation.
        config_path (Path): Path to configuration used for evaluation.
        ckpt (Path): Model checkpoint used for evaluation.
        log_path (str): Path to log directory.
    """
    n_episodes = config["evaluation_num_episodes"]
    results_path = Path(log_path)
    if not results_path.exists():
        results_path.mkdir()
    results_path /= f"{config_path.stem}_{ckpt.name}_{n_episodes}_episodes_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

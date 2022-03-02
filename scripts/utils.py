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


import argparse
import subprocess
from argparse import Namespace
from typing import Any, Dict

import numpy as np
import yaml
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import grid_search
from tesse.msgs import Camera
from tesse_gym import ObservationConfig, get_network_config
from tesse_gym.core.utils import set_all_camera_params
from tesse_gym.observers.dsg.scene_graph import get_scene_graph_task
from tesse_gym.tasks.goseek import GoSeek


def populate_rllib_config(
    default_config: Dict[str, Any], user_config: Dict[str, str]
) -> Dict[str, Any]:
    """Add or edit items from an rllib config.
    If `user_config` contains a value wrapped in the string
    `grid_search([])`, the value will be given as and
    rllib.tune `grid_search` option.
    Args:
        default_config (Dict[str, Any]): A default rllib config
            (e.g., for ppo).
        user_config (Dict[str, str]): Configuration given by user.
    Returns:
        Dict[str, Any]: `default_config` with `user_config` items
            added. If there are matching keys, `user_config` takes
            precedence.
    """
    with open(user_config) as f:
        user_config = yaml.load(f)

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
            lambda x: x.observers["dsg"].get_visited_nodes()
        )
        episode.custom_metrics["found_targets"] = mean_found_targets
        episode.custom_metrics["collisions"] = mean_collisions
        episode.custom_metrics["visited_nodes"] = n_visited_nodes

        if hasattr(base_env.get_unwrapped()[0], "visited_cells"):
            mean_visited_cells = _get_unwrapped_env_value(
                lambda x: len(x.visited_cells)
            )
            episode.custom_metrics["n_visited_cells"] = mean_visited_cells


def init_function(env):
    # TODO config
    cnn_shape = (5, 120, 160)
    set_all_camera_params(
        env, height_in_pixels=cnn_shape[1], width_in_pixels=cnn_shape[2]
    )


def make_goseek_env(config):
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


def timesteps_to_train_itrs(batch_size, save_freq_timesteps):
    """Get save frequency in training iterations"""
    return save_freq_timesteps // batch_size


def get_args() -> Namespace:
    """Get arguments for DSG-RL training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--name")
    parser.add_argument("--timesteps", default=5000000, type=int)
    parser.add_argument("--sim_ok", action="store_true")
    parser.add_argument("--restore", default=None)
    return parser.parse_args()

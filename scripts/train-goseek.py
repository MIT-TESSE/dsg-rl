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


import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from rllib_policies.gnn import ActionLayerGNNActorCritic
from tesse.msgs import Camera
from tesse_gym import ObservationConfig, get_network_config
from tesse_gym.core.utils import set_all_camera_params
from tesse_gym.observers.dsg.scene_graph import get_scene_graph_task

# from tesse_gym.rllib.networks import NatureCNNActorCritic, NatureCNNRNNActorCritic
from tesse_gym.rllib.utils import (
    check_for_tesse_instances,
    get_args,
    populate_rllib_config,
)
from tesse_gym.tasks.goseek import GoSeek


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
    set_all_camera_params(
        env, height_in_pixels=cnn_shape[1], width_in_pixels=cnn_shape[2]
    )


def make_goseek_env(config):
    modalities = []
    cnn_channels = 0
    if "RGB" in config["modalities"]:
        modalities.append(Camera.RGB_LEFT)
        cnn_channels += 3
    if "SEGMENTATION" in config["modalities"]:
        modalities.append(Camera.SEGMENTATION)
        cnn_channels += 1
    if "DEPTH" in config["modalities"]:
        modalities.append(Camera.DEPTH)
        cnn_channels += 1
    # assert cnn_channels == cnn_shape[0]

    observation_config = ObservationConfig(
        modalities=modalities,
        height=cnn_shape[1],
        width=cnn_shape[2],
        pose=True,
        use_dict=True,
    )

    print(observation_config)

    worker_index = config.worker_index
    vector_index = config.vector_index
    N_ENVS_PER_WORKER = 1
    print(f"Worker / rank: {worker_index}, {vector_index}")

    rank = worker_index - 1 + (N_ENVS_PER_WORKER + 1) * vector_index
    scene = config["SCENES"][rank % len(config["SCENES"])]

    if "RANK" in config.keys():
        config_rank = config.pop("RANK")
        rank = config_rank if isinstance(config_rank, int) else config_rank[rank]

    print(
        f"MAKING GOSEEK ENV w/ rank: {rank}, inds: ({worker_index}, {vector_index}, scene: {scene})"
    )

    # if "esdf_data" not in config.keys():
    #     config["esdf_data"] = None
    # if "coordinate_system" not in config.keys():
    #     config["coordinate_system"] = "cartesian"
    # if (
    #     "agent_frame_type" in config.keys()
    #     and "agent_frame_edge_filter_ref" not in config.keys()
    # ):
    # config["agent_frame_edge_filter_ref"] = "layer_node"

    config.pop("SCENES")
    config.pop("modalities")

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


if __name__ == "__main__":
    args = get_args()
    if not args.sim_ok:
        check_for_tesse_instances()

    ray.init()

    cnn_shape = (5, 120, 160)
    ModelCatalog.register_custom_model("gnn_actor_critic", ActionLayerGNNActorCritic)
    register_env("goseek", make_goseek_env)

    config = ppo.DEFAULT_CONFIG.copy()
    config["callbacks"] = GOSEEKGoalCallbacks
    config = populate_rllib_config(config, args.config)
    config["env_config"]["video_log_path"] = (
        config["env_config"]["video_log_path"] + f"/{args.name}"
    )
    local_dir = config.pop("local_dir")
    save_freq_timesteps = config.pop("ckpt_save_freq_timesteps")
    save_freq_itrs = timesteps_to_train_itrs(
        config["train_batch_size"], save_freq_timesteps
    )
    config["evaluation_interval"] = save_freq_itrs

    search_exp = tune.Experiment(
        name=args.name,
        run="PPO",
        config=config,
        stop={"timesteps_total": args.timesteps},
        checkpoint_freq=save_freq_itrs,
        checkpoint_at_end=True,
        local_dir=local_dir,
        restore=args.restore,
    )

    tune.run_experiments([search_exp])

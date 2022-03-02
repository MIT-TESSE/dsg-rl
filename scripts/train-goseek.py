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


import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rllib_policies.gnn import ActionLayerGNNActorCritic

from utils import (
    GOSEEKGoalCallbacks,
    check_for_tesse_instances,
    get_args,
    make_goseek_env,
    populate_rllib_config,
    timesteps_to_train_itrs,
)

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

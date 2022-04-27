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
from argparse import Namespace

import ray
from dsg_rl import check_for_tesse_instances, get_ppo_train_config, make_goseek_env
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rllib_policies.gnn import ActionLayerGNNActorCritic


def get_args() -> Namespace:
    """Get arguments for DSG-RL training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training configuration file.")
    parser.add_argument("--name", type=str, help="Experiment log directory name")
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Number of simulation timesteps for which to train.",
    )
    parser.add_argument(
        "--sim-ok",
        action="store_true",
        help="Start training even if there are other simulator instances running. "
        "If true, be sure that simulation ports don't conflict.",
    )
    parser.add_argument(
        "--restore",
        default=None,
        type=str,
        help="Restore training from this checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not args.sim_ok:
        check_for_tesse_instances()

    # ray initialization
    ray.init()
    cnn_shape = (5, 120, 160)
    ModelCatalog.register_custom_model("gnn_actor_critic", ActionLayerGNNActorCritic)
    register_env("goseek", make_goseek_env)

    # get configuration and train
    config, local_dir = get_ppo_train_config(args.config, args.name)

    search_exp = tune.Experiment(
        name=args.name,
        run="PPO",
        config=config,
        stop={"timesteps_total": args.timesteps},
        checkpoint_freq=config["evaluation_interval"],
        checkpoint_at_end=True,
        local_dir=local_dir,
        restore=args.restore,
    )

    tune.run_experiments([search_exp])

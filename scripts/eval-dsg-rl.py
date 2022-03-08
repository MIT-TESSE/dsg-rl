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
import pprint
from pathlib import Path

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rllib_policies.gnn import ActionLayerGNNActorCritic

from utils import (
    check_for_tesse_instances,
    get_ppo_eval_config,
    log_eval_results,
    make_goseek_env,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Agent configuration")
    parser.add_argument("--ckpt", help="Checkpoint to evaluate.")
    parser.add_argument("--episodes", help="Number of episodes to run", default=None)
    parser.add_argument(
        "--sim-ok",
        action="store_true",
        help="Start training even if there are other simulator instances running. "
        "If true, be sure that simulation ports don't conflict.",
    )
    parser.add_argument(
        "--log-path", type=str, help="Name of evaluation result directory. "
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
    register_env("goseek", make_goseek_env)

    # setup configuration
    config_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    config = get_ppo_eval_config(args.config, ckpt_path, args.log_path, args.episodes)
    local_dir = config.pop("local_dir")

    # load trainer
    trainer = ppo.PPOTrainer(config)
    trainer.restore(args.ckpt)

    # run evaluation
    results = trainer.evaluate()
    pprint.pprint(results)
    log_eval_results(results, config, config_path, ckpt_path, args.log_path)

    ray.shutdown()

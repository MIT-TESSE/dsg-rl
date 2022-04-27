---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.4
  kernelspec:
    display_name: Python [conda env:test-tg]
    language: python
    name: conda-env-test-tg-py
---

## Notebook to run DSG-RL agent and plot example episodes


```python
import yaml
import ray
from pathlib import Path

%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.animation as animation

%load_ext autoreload
%autoreload 2
from tesse_gym.observers.dsg.scene_graph import get_scene_graph_task
from tesse_gym.core.utils import set_all_camera_params
from tesse_gym.tasks.goseek import GoSeek
from tesse_gym import ObservationConfig, get_network_config
from rllib_policies.gnn import ActionLayerGNNActorCritic
from dsg_rl import get_ppo_eval_config, make_goseek_env

from tesse.msgs import Camera

from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo
```

## Set paths for:

-ckpt_path: path to model checkpoint

-config_path: path to ppo / simulation config

```python
ckpt_path = "CKPT_PATH"
config_path = "CONFIG_PATH"
```

## Restart Ray

```python
ray.shutdown()
ray.init()
```

## Start PPO agent

```python
!ps aux | grep goseek  # check for any running sim instances 
```

```python
if "trainer" in globals():
    trainer.stop()
    env.close()
```

```python
ModelCatalog.register_custom_model("gnn_actor_critic", ActionLayerGNNActorCritic)

register_env("goseek", make_goseek_env)

ppo_config = get_ppo_eval_config(Path(config_path), ckpt=Path(ckpt_path), log_path="./eval_logs")

trainer = ppo.PPOTrainer(ppo_config)
trainer.load_checkpoint(ckpt_path)
```

## Start Simulator

```python
if "env" in globals():
    env.close()
```

```python
# setup configuration
with open(config_path) as f:
    config = yaml.load(f)
    
config["env_config"]["query_image_data"] = True  # for visualization

rank = 0
scene = 1

config["env_config"]["build_path"] = config["env_config"].pop("FILENAME")
config["env_config"]["scene_id"] = config["env_config"].pop("SCENES")

observation_config = ObservationConfig(
    modalities=(Camera.RGB_LEFT, Camera.SEGMENTATION, Camera.DEPTH),
    height=120,
    width=160,
    pose=True,
    use_dict=True,
)

_ = config["env_config"].pop("modalities", "")

# start simulation
env = get_scene_graph_task(
    GoSeek,
    network_config=get_network_config(
        simulation_ip="localhost", own_ip="localhost", worker_id=rank
    ),
    observation_config=observation_config,
    init_function=lambda env: set_all_camera_params(
        env, height_in_pixels=120, width_in_pixels=160
    ),
    **config["env_config"],
)
```

## Reset episode
also plot image from observation

```python
obs = env.reset(1)
```

```python
fig, ax = plt.subplots()
ax.imshow(obs["RGB_LEFT"])
```

## Run agent for _n_ steps
The full episode length is 400

```python
frames = []
```

```python
plot_real_time = False  # real time plotting accumulates high latency
n_steps = 50 # full episode is 400 steps


for i in range(n_steps):
    actions = trainer.compute_actions({"default_policy": obs})
    obs, _, _, _ = env.step(actions["default_policy"])
    frames.append(obs["RGB_LEFT"])

    if plot_real_time:
        ax.imshow(frames[-1], animated=True)
        fig.canvas.draw()
```

## Visualize agent's run

```python
fig, ax = plt.subplots()
im = plt.imshow(frames[1], interpolation="none", aspect="auto", vmin=0, vmax=1)


def animate_func(i):
    im.set_array(frames[i])
    return [im]

anim = animation.FuncAnimation(
    fig,
    animate_func,
    frames=len(frames),
    interval=100,  # in ms
)
```

## Shutdown agent and simulatior

```python
trainer.stop()
env.close()
```

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and
Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

(c) 2020 Massachusetts Institute of Technology.

MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
than as specifically authorized by the U.S. Government may violate any copyrights that exist in
this work.


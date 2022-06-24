# dsg-rl

Code for the paper 
[*Hierarchical Representation and Explicit Memory: Learning Effective Navigation Polices on 3D Scene Graphs using Graph Neural Networks*](https://arxiv.org/abs/2108.01176)
published at the International Conference on Robotics and Automation (ICRA) 2022.


# Installation 

We reccomend installing in a virtual environment, such as [Anaconda](https://www.anaconda.com/) or [venv](https://docs.python.org/3/tutorial/venv.html). We use Python 3.7 in our work. 

Clone and install this repo:
```
git clone git@github.mit.edu:TESS/dsg-rl.git
cd dsg-rl
python -m pip install .
```

Next, install [Pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html).

Pytorch geometric is sensitive to PyTorch and CUDA versions, so we recommend that you install this dependency by following the author's [instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 
You can check your PyTorch and CUDA versions via the following command
```python
>> python
>> import torch
>> torch.__version__
1.10
>> torch.version.cuda
11.3
```

For example, installing via pip with PyTorch version 1.10 and CUDA 11.3 would look like:
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```


# Running

## Configure experiment

See the provided [train-template.yaml](./config/train-template.yaml) for an example configuration file. 
Note that any fields in <BRACKETS> will need to be updated with the correct device-specific path.
In particualr, paths to the TESSE simulator, Dynamic Scene Graphs (DSGs), and Euclidean Signed Distance Functions (EDSFs) must be provided,
all of which may be found [here](https://github.com/MIT-TESSE/dsg-rl/releases/tag/0.1.0).

## Training 

The [`train-dsg-rl.py`](./scripts/train-dsg-rl.py) script may be used to train an agent.

```
>>> python ./scripts/train-dsg-rl.py -h
usage: train-dsg-rl.py [-h] [--config CONFIG] [--name NAME]
                       [--timesteps TIMESTEPS] [--sim-ok] [--restore RESTORE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Training configuration file.
  --name NAME           Experiment log directory name
  --timesteps TIMESTEPS
                        Number of simulation timesteps for which to train.
  --sim-ok              Start training even if there are other simulator
                        instances running. If true, be sure that simulation
                        ports don't conflict.
  --restore RESTORE     Restore training from this checkpoint
```

## Evaluation 

The [eval-dsg-rl.py](./scripts/eval-dsg-rl.py) may be used to evaluate and agent.

```
>> python ./scripts/eval-dsg-rl.py -h
usage: eval-dsg-rl.py [-h] [--config CONFIG] [--ckpt CKPT]
                      [--episodes EPISODES] [--sim-ok] [--log-path LOG_PATH]

optional arguments:
  -h, --help           show this help message and exit
  --config CONFIG      Agent configuration
  --ckpt CKPT          Checkpoint to evaluate.
  --episodes EPISODES  Number of episodes to run
  --sim-ok             Start training even if there are other simulator
                       instances running. If true, be sure that simulation
                       ports don't conflict.
  --log-path LOG_PATH  Name of evaluation result directory.
```

# Disclaimer

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

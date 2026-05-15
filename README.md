# ADM-v2: Any-step Dynamics Model v2

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HxLyn3/adm2/blob/main/LICENSE)

This is the code for the paper [ADM-v2: Pursuing Full-Horizon Roll-out in Dynamics Models for Offline Policy Learning and Evaluation](https://openreview.net/forum?id=ICbXEwqpga) in ICLR 2026.

## Requirements

```
cd ~
git clone https://github.com/th1991-01/adm2
cd adm2

pyenv local 3.10
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2   --index-url https://download.pytorch.org/whl/cu118

cd ../
git clone https://github.com/Polixir/neorl
cd neorl
pip install -e .
pip install numpy==1.26.4
```

To install all the required dependencies:

1. Install MuJoCo engine, which can be downloaded from [here](https://mujoco.org/download).
2. Install Python packages listed in `requirements.txt` using `pip install -r requirements.txt`. You should specify the version of `mujoco-py` in `requirements.txt` depending on the version of MuJoCo engine you have installed.
3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl).
4. Manually download and install `neorl` package from [here](https://github.com/polixir/NeoRL).

## Run an experiment 

```shell
python main.py --env [Env] --env-name [Env name] 
```

The config files act as defaults for a task. They are all located in `config`. `--env` refers to the benchmark, D4RL or NeoRL. `--env-name` refers to the config files in `config/`. All results will be stored in the `result` folder.

For example, run ADM-v2 for policy optimization on hopper-medium-v2 dataset of D4RL benchmark:

```bash
python main.py --env d4rl --env-name hopper-medium-v2
```

## Citation
If you find this repository useful for your research, please cite:
```bash
@inproceedings{
    adm2,
    author       = {Haoxin Lin and
                    Siyuan Xiao and
                    Yi-Chen Li and
                    Zhilong Zhang and
                    Yihao Sun and
                    Chengxing Jia and
                    Yang Yu},
    title        = {ADM-v2: Pursuing Full-Horizon Roll-out in Dynamics Models for Offline Policy Learning and Evaluation},
    booktitle    = {The 14th International Conference on Learning Representations (ICLR'26)},
    year         = {2026},
    address      = {Rio de Janeiro, Brazil}
}
```

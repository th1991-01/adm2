import os
import copy
import yaml
import random
import argparse
import setproctitle

import torch
import numpy as np

from runner.ope_runner import OPERunner

def get_args():
    parser = argparse.ArgumentParser(description="Model as Simulator")

    # environment settings
    parser.add_argument("--env", type=str, default="d4rl")
    parser.add_argument("--env-name", type=str, default="hopper-medium-v2")
    
    # dynamics model parameters
    # "adm2" -- our proposed any-step dynamics model (ADM-v2)
    # "adm"  -- original any-step dynamics model
    # "en"   -- ensemble dynamics model
    # "rnn"  -- RNN dynamics model
    # "dreamer" -- Dreamer dynamics model
    parser.add_argument("--dyna-model", type=str, default="adm2", choices=["adm2", "adm", "en", "rnn", "dreamer"])
    parser.add_argument("--model-hidden-dim", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=1000)
    parser.add_argument("--given-reward", type=bool, default=False)
    parser.add_argument("--load-label", type=str, default="adm2-sac")
    parser.add_argument("--load-time", type=str, default="25-0629-102410")
    parser.add_argument("--load-seed", type=int, default=0)
    parser.add_argument("--max-adm-step", type=int, default=5)                          # maximum length of rnn input
    # only for adm2
    parser.add_argument("--n-starts", type=int, default=5)
    
    parser.add_argument("--n-trajs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--regret-k", type=int, default=1)

    # running parameters
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    """ main function """
    args = get_args()
    algo_yml_path = "./config/{}/{}.yml".format(args.env, args.env_name.split("-v")[0])
    algo_yml = yaml.load(open(algo_yml_path, 'r'), Loader=yaml.FullLoader)
    for key, value in algo_yml.items():
        setattr(args, key, value)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # set seed of torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    runner = OPERunner(copy.deepcopy(args))
    runner.run()

if __name__ == "__main__":
    main()
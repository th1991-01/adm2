import os
import copy
import yaml
import random
import argparse
import setproctitle

import torch
import numpy as np

from runner.mas_trainer import ModelSimTrainer

def get_args():
    parser = argparse.ArgumentParser(description="Model as Simulator")

    # environment settings
    parser.add_argument("--env", type=str, default="d4rl")
    parser.add_argument("--env-name", type=str, default="walker2d-medium-v2")
    
    # dynamics model parameters
    # "adm2" -- our proposed any-step dynamics model (ADM-v2)
    # "adm"  -- original any-step dynamics model
    # "en"   -- ensemble dynamics model
    # "rnn"  -- RNN dynamics model
    # "dreamer" -- Dreamer dynamics model
    parser.add_argument("--dyna-model", type=str, default="adm2", choices=["adm2", "adm", "en", "rnn", "dreamer"])
    parser.add_argument("--model-hidden-dim", type=int, default=200)
    parser.add_argument("--model-lr", type=float, default=3e-4)
    parser.add_argument("--rollout-batch-size", type=int, default=4096)
    parser.add_argument("--rollout-length", type=int, default=1000)
    parser.add_argument("--given-reward", type=bool, default=False)
    parser.add_argument("--load-model", type=bool, default=False)
    parser.add_argument("--load-label", type=str, default="adm2-sac")
    parser.add_argument("--load-time", type=str, default="25-0531-001907")
    parser.add_argument("--load-seed", type=int, default=0)
    parser.add_argument("--max-adm-step", type=int, default=5)                          # maximum length of rnn input
    # only for adm2
    parser.add_argument("--n-starts", type=int, default=5)

    # policy optimization parameters
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "td3"])
    parser.add_argument("--ac-hidden-dims", type=list, default=[256, 256])              # dimensions of actor/critic hidden layers
    parser.add_argument("--actor-lr", type=float, default=1e-4)                         # learning rate of actor
    parser.add_argument("--lr-schedule", type=bool, default=True)
    parser.add_argument("--critic-lr", type=float, default=3e-4)                        # learning rate of critic
    parser.add_argument("--gamma", type=float, default=0.99)                            # discount factor
    parser.add_argument("--tau", type=float, default=0.005)                             # update rate of target network
    parser.add_argument("--penalty-coef", type=float, default=5)                        # penalty coefficient
    parser.add_argument("--real-ratio", type=float, default=0.05)
    # for SAC
    parser.add_argument("--alpha", type=float, default=0.05)                            # weight of entropy
    parser.add_argument("--auto-alpha", type=bool, default=True)                        # auto alpha adjustment
    parser.add_argument("--alpha-lr", type=float, default=1e-4)                         # learning rate of alpha
    parser.add_argument("--target-entropy", type=int, default=None)                     # target entropy
    parser.add_argument("--deterministic-backup", type=bool, default=False)
    parser.add_argument("--q-clip", type=float, default=None)
    # for TD3
    parser.add_argument("--explore-noise", type=float, default=0.25)
    parser.add_argument("--target-policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    # for PPO
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-clip", type=bool, default=True)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ppo-slice", type=int, default=24)
    parser.add_argument("--ppo-epoch", type=int, default=5)

    # replay-buffer parameters (for off-policy algos: sac, td3)
    parser.add_argument("--buffer-size", type=int, default=int(1e7))

    # running parameters
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=2000)
    parser.add_argument("--step-per-epoch", type=int, default=25)
    parser.add_argument("--updates-per-step", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)                          # mini-batch size
    parser.add_argument("--eval-n-episodes", type=int, default=10)
    parser.add_argument("--test-n-episodes", type=int, default=int(1e3))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs='*', default=list(range(10)))

    args = parser.parse_args()
    return args

def main():
    """ main function """
    args = get_args()
    algo_yml_path = "./config/{}/{}.yml".format(args.env, args.env_name.split("-v")[0])
    algo_yml = yaml.load(open(algo_yml_path, 'r'), Loader=yaml.FullLoader)
    for key, value in algo_yml.items():
        setattr(args, key, value)

    setproctitle.setproctitle("{} {}".format(args.algo.upper(), args.env_name))

    for seed in args.seeds:
        args.seed = seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

        # set seed of torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        runner = ModelSimTrainer(copy.deepcopy(args))
        runner.run()
        del runner

if __name__ == "__main__":
    main()
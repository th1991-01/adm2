import os
import gym
import d4rl
import json
import argparse

import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Model as Simulator")

    # environment settings
    parser.add_argument("--env", type=str, default="d4rl")
    parser.add_argument("--env-name", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--load-label", type=str, default="sadm-sac")
    parser.add_argument("--load-times", type=str, nargs="+", default=None)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    res_path = f"./result/{args.env}/{args.env_name}/{args.load_label}"

    if args.load_times is None:
        args.load_times = os.listdir(res_path)
    
    scores = []
    print(f"task num: {len(args.load_times)}")
    for load_time in args.load_times:
        rec_path = f"{res_path}/{load_time}/record"
        for fname in os.listdir(rec_path):
            if fname.split('.')[1] != "txt": continue
            if "model" in fname: continue
            f_path = os.path.join(rec_path, fname)
            with open(f_path, "r") as f:
                record = json.load(f)
                if "score_mean" in record:
                    scores.append(record["score_mean"][-1])
                    
    print(f"task: {args.env_name}@{args.env}")
    print(f"score mean: {np.mean(scores)}")
    print(f"score std: {np.std(scores)}")

if __name__ == "__main__":
    main()
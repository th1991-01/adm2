import os
import gym
import json
import d4rl
import neorl
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class BASETrainer:
    """ base trainer """

    def __init__(self, args):
        if args.env == "neorl":
            self.make_env = lambda env_name: neorl.make(env_name)
        else:
            self.make_env = lambda env_name: gym.make(env_name)
            
        # init env
        self.env = self.make_env(args.env_name)
        self.env.action_space.seed(args.seed)

        self.eval_env = self.make_env(args.env_name)
        self.eval_env.action_space.seed(args.seed)

        if args.env == "adroit" or args.env == "maze":
            self.env.seed(args.seed)
            self.eval_env.seed(args.seed)
        else:
            self.env.reset(seed=args.seed)
            self.eval_env.reset(seed=args.seed)

        args.obs_shape = self.env.observation_space.shape
        args.action_space = self.env.action_space
        args.action_dim = np.prod(args.action_space.shape)

        # running parameters
        self.batch_size = args.batch_size
        self.eval_n_episodes = args.eval_n_episodes
        self.device = args.device
        self.seed = args.seed
        self.args = args

        dtime = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
        if args.env == "neorl": args.env_name += f"-{args.data_type}"
        self.model_dir = f"./result/{args.env}/{args.env_name}/{args.dyna_model}-{args.algo}/{dtime}/model"
        self.record_dir = f"./result/{args.env}/{args.env_name}/{args.dyna_model}-{args.algo}/{dtime}/record"
        self.log_dir = f"./result/{args.env}/{args.env_name}/{args.dyna_model}-{args.algo}/{dtime}/log"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.load_model = args.load_model
        self.load_time = args.load_time
        self.load_seed = args.load_seed
        if self.load_model:
            self.load_dir = f"./result/{args.env}/{args.env_name}/{args.load_label}/{self.load_time}/model"
        else:
            self.load_dir = None

        self.logger = SummaryWriter(self.log_dir)

    def _eval_policy(self):
        """ evaluate policy """
        episode_rewards = []
        for _ in range(self.eval_n_episodes):
            done = False
            episode_rewards.append(0)
            obs = self.eval_env.reset()
            while not done:
                action = self.agent.act(obs, deterministic=True).cpu().numpy()
                obs, reward, done, _ = self.eval_env.step(action)
                episode_rewards[-1] += reward
        return episode_rewards

    def _save(self, records):
        """ save model and record """
        self.dyna_model.save_model(os.path.join(self.model_dir, "dyna_seed-{}.pth".format(self.seed)))
        self.agent.save_model(os.path.join(self.model_dir, "agent_seed-{}.pth".format(self.seed)))
        with open(os.path.join(self.record_dir, "record_seed-{}.txt".format(self.seed)), "w") as f:
            json.dump(records, f)

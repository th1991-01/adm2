import os
import gym
import time
import copy
import json
import d4rl
import neorl
import torch
import numpy as np
from tqdm import tqdm

from dynamics.adm_dynamics import ADMDynamics
from dynamics.sadm_dynamics import SADMDynamics
from dynamics.ensemble_dynamics import EnsembleDynamics
from dynamics.rnn_dynamics import RNNDynamics
from dynamics.dreamer_dynamics import DreamerDynamics
from components.static_fns import STATICFUNC
from env.model_as_sim import ADMSim, SADMSim, EnSim, RNNSim
from buffer.buffer4seqsamp import ReplayBufferForSeqSampling
from dope_policies import DOPE_POLICY_PATH, DOPEPolicy


class OPERunner:
    """ off-policy evaluation """

    def __init__(self, args):
        if args.env == "neorl":
            task, data_type, version = tuple(args.env_name.split('-'))
            args.env_name = task + '-' + version
            args.data_type = data_type
        self.args_dict = copy.deepcopy(vars(args))
        
        if args.env == "neorl":
            self.make_env = lambda env_name: neorl.make(env_name)
        else:
            self.make_env = lambda env_name: gym.make(env_name)
            
        # init env
        self.env = self.make_env(args.env_name)
        self.env.action_space.seed(args.seed)

        if args.env == "adroit" or args.env == "maze":
            self.env.seed(args.seed)
        else:
            self.env.reset(seed=args.seed)

        args.obs_shape = self.env.observation_space.shape
        args.action_space = self.env.action_space
        args.action_dim = np.prod(args.action_space.shape)
        
        self.task = args.env_name.split('-')[0].lower()
        self.dope_policy_dir = os.path.join(DOPE_POLICY_PATH, self.task)
        self.dope_policy_pkls = [os.path.join(self.dope_policy_dir, filename) for filename in os.listdir(self.dope_policy_dir)]
        self.dope_policies = [DOPEPolicy(pkl, args.device) for pkl in self.dope_policy_pkls]

        if args.env == "neorl": args.env_name += f"-{args.data_type}"
        self.load_time = args.load_time
        self.load_seed = args.load_seed
        self.record_dir = f"./result/{args.env}/{args.env_name}/{args.load_label}/{self.load_time}/record"

        # init dynamics model
        task = args.env_name.split('-')[0]
        if args.env == "neorl": task = "neorl-" + task
        if args.env == "maze": task = task + "-" + args.env_name.split('-')[1]
        self.static_fn = STATICFUNC[task.lower()]
        if args.dyna_model == "adm":
            self.dyna_model = ADMDynamics(
                obs_dim=np.prod(args.obs_shape),
                action_dim=args.action_dim,
                hidden_dim=args.model_hidden_dim,
                max_adm_step=args.max_adm_step,
                device=args.device
            )
            self.ModelSim = ADMSim
        elif args.dyna_model == "sadm":
            self.dyna_model = SADMDynamics(
                obs_dim=np.prod(args.obs_shape),
                action_dim=args.action_dim,
                hidden_dim=args.model_hidden_dim,
                max_adm_step=args.max_adm_step,
                device=args.device
            )
            self.ModelSim = SADMSim
        elif args.dyna_model == "en":
            self.dyna_model = EnsembleDynamics(
                obs_dim=np.prod(args.obs_shape),
                action_dim=args.action_dim,
                device=args.device
            )
            self.ModelSim = EnSim
        elif args.dyna_model == "rnn":
            self.dyna_model = RNNDynamics(
                obs_dim=np.prod(args.obs_shape),
                action_dim=args.action_dim,
                hidden_dim=args.model_hidden_dim,
                max_adm_step=args.max_adm_step,
                device=args.device
            )
            self.ModelSim = RNNSim
        elif args.dyna_model == "dreamer":
            self.dyna_model = DreamerDynamics(
                obs_dim=np.prod(args.obs_shape),
                action_dim=args.action_dim,
                hidden_dim=args.model_hidden_dim,
                max_adm_step=args.max_adm_step,
                device=args.device
            )
            self.ModelSim = RNNSim
            
        # load dynamics model
        self.load_dir = f"./result/{args.env}/{args.env_name}/{args.load_label}/{self.load_time}/model"
        load_path = os.path.join(self.load_dir, "dyna_seed-{}.pth".format(self.load_seed))
        state_dict = torch.load(load_path)
        self.dyna_model.load_state_dict(state_dict)

        # init replay buffer to store environmental data
        self.dataset = ReplayBufferForSeqSampling(
            buffer_size=1000000,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim
        )
        rew_bias = 1 if args.env == "maze" else 0
        if args.env == "neorl":
            dataset, _ = self.env.get_dataset(data_type=args.data_type, train_num=1000, need_val=False)
            self.dataset.load_neorl_dataset(dataset, rew_bias)
        else:
            self.dataset.load_dataset(self.env.get_dataset(), rew_bias)

        # other parameters
        self.max_adm_step = args.max_adm_step
        self.n_starts = min(args.max_adm_step, args.n_starts)
        self.rollout_length = args.rollout_length
        self.given_reward = args.given_reward
        self.device = args.device
        self.seed = args.seed
        
        self.n_trajs = args.n_trajs
        self.gamma = args.gamma
        self.regret_k = args.regret_k
        
    def run(self):
        # build model-based env
        eval_init_seqs = self.dataset.sample_all_head_nstep(self.n_starts-1)
        eval_init_seqs["s"] = torch.cat((eval_init_seqs["s"], eval_init_seqs["s_"][:, -1:]), dim=1)
        self.eval_model_env = self.ModelSim(
            dynamics=copy.deepcopy(self.dyna_model),
            static_fn=self.static_fn,
            max_steps=self.rollout_length,
            init_obs_seqs=eval_init_seqs["s"],
            init_act_seqs=eval_init_seqs["a"],
            n_parallels=self.n_trajs,
            given_reward=self.given_reward
        )
        
        # init records
        records = {
            "model_evaluation": [],
            "real_env_evaluation": [],
            "raw_mae": None,
            "norm_mae": None,
            "rank_corr": None,
            f"regret@{self.regret_k}": None
        }
            
        # roll-out in model
        for cnt in tqdm(range(len(self.dope_policies)), desc="Roll-out in Model"):
            policy = self.dope_policies[cnt]
            model_obs = self.eval_model_env.reset_all()
            done = torch.tensor([False]*model_obs.shape[0], dtype=torch.bool, device=self.device)
            episode_reward = torch.tensor([0]*model_obs.shape[0], dtype=torch.float32, device=self.device)
            for t in range(self.rollout_length):
                if not done.all():
                    action = policy.select_action(model_obs, deterministic=True)
                    action_torch = torch.as_tensor(action, dtype=torch.float32, device=self.device)
                    model_obs, reward, _, terminated, truncated = self.eval_model_env.step(action_torch)
                    episode_reward[~done] += self.gamma**t * reward.flatten()[~done]
                    done[terminated.flatten() | truncated.flatten()] = True
            records["model_evaluation"].append(float(episode_reward.mean().item()))
        
        # roll-out in model
        for cnt in tqdm(range(len(self.dope_policies)), desc="Roll-out in Real Env"):
            policy = self.dope_policies[cnt]
            episode_rewards = self._eval_policy(policy)
            records["real_env_evaluation"].append(float(np.mean(episode_rewards)))
            
        model_values = np.array(records["model_evaluation"])
        real_values = np.array(records["real_env_evaluation"])
        value_min, value_max = real_values.min(), real_values.max()
        norm_model_values = (model_values - value_min) / (value_max - value_min)
        norm_real_values = (real_values - value_min) / (value_max - value_min)
        
        top_ids = np.argsort(norm_model_values)[-self.regret_k:]
        regret = norm_real_values.max() - norm_real_values[top_ids].max()
        
        records["raw_mae"] = np.abs(model_values - real_values).mean()
        records["norm_mae"] = np.abs(norm_model_values - norm_real_values).mean()
        records["rank_corr"] = np.corrcoef(norm_real_values, norm_model_values)[0, 1]
        records[f"regret@{self.regret_k}"] = regret
        
        print(f"raw absolute error: {records['raw_mae']}")
        print(f"normalized absolute error: {records['norm_mae']}")
        print(f"rank correlation: {records['rank_corr']}")
        print(f"regret@{self.regret_k}: {records[f'regret@{self.regret_k}']}")
            
        with open(os.path.join(self.record_dir, "ope_seed-{}.txt".format(self.load_seed)), "w") as f:
            json.dump(records, f)
            
    def _eval_policy(self, policy):
        """ evaluate policy """
        episode_rewards = []
        for _ in range(self.n_trajs):
            done = False
            episode_rewards.append(0)
            obs = self.env.reset()
            t = 0
            while not done:
                action = policy.select_action(obs, deterministic=True).flatten()
                obs, reward, done, _ = self.env.step(action)
                episode_rewards[-1] += self.gamma**t * reward
                t += 1
        return episode_rewards

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

from dynamics.mujoco_oracle_dynamics import MujocoOracleDynamics
from dynamics.adm_dynamics import ADMDynamics
from dynamics.sadm_dynamics import SADMDynamics
from components.static_fns import STATICFUNC
from env.model_as_sim import ADMSim, SADMSim
from buffer.buffer4seqsamp import ReplayBufferForSeqSampling
from dope_policies import DOPE_POLICY_PATH, DOPEPolicy


class RolloutEvaluator:
    """ model-rollout evaluator """

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
        self.rollout_batch_size = 100
        self.rollout_length = args.rollout_length
        self.device = args.device
        self.seed = args.seed
        
    def run(self):
        # mujoco env
        self.mujoco_env = MujocoOracleDynamics(self.env)
        
        # build model-based env
        eval_init_seqs = self.dataset.sample_all_head_nstep(self.n_starts-1)
        eval_init_seqs["s"] = torch.cat((eval_init_seqs["s"], eval_init_seqs["s_"][:, -1:]), dim=1)
        self.eval_model_env = self.ModelSim(
            dynamics=copy.deepcopy(self.dyna_model),
            static_fn=self.static_fn,
            max_steps=self.rollout_length,
            init_obs_seqs=eval_init_seqs["s"],
            init_act_seqs=eval_init_seqs["a"],
            n_parallels=self.rollout_batch_size
        )
            
        # rollout policy
        policy = self.dope_policies[3]
        
        records = {
            "rollout_length": [],
            "prediction_error_mean": [],
            "prediction_error_std": [],
            "prediction_error_max": [],
            "prediction_error_min": []
        }
        
        cnts = [0] * self.rollout_length
        errors = [0] * self.rollout_length
        rollout_times = []
            
        # roll-out in model env
        init_obs = self.eval_model_env.reset_all()
        model_obs = copy.deepcopy(init_obs).cpu().numpy()
        mujoco_obs = copy.deepcopy(init_obs).cpu().numpy()
        done = torch.tensor([False]*model_obs.shape[0])
        pbar = tqdm(range(self.rollout_length), desc="Roll-out in ModelEnv")
        for step in pbar:
            if not done.all():
                action = policy.select_action(model_obs, deterministic=True)
                action_torch = torch.as_tensor(action, dtype=torch.float32, device=self.device)
                start = time.time()
                model_obs, _, _, terminated, truncated = self.eval_model_env.step(action_torch)
                end = time.time()
                rollout_times.append((end - start)*1000)
                done[terminated.flatten() | truncated.flatten()] = True
                
                for id in range(mujoco_obs.shape[0]):
                    mujoco_obs[id], _, _, _ = self.mujoco_env.step(mujoco_obs[id], action[id])
                    
                if not done.all():
                    rollout_error = np.mean(np.square(model_obs.cpu().numpy() - mujoco_obs), axis=-1)
                    rollout_error = rollout_error[~done]
                    rollout_error_mean = np.mean(rollout_error)
                    
                    records["rollout_length"].append(step + 1)
                    records["prediction_error_mean"].append(float(rollout_error_mean))
                    records["prediction_error_std"].append(float(np.std(rollout_error)))
                    records["prediction_error_max"].append(float(np.max(rollout_error)))
                    records["prediction_error_min"].append(float(np.min(rollout_error)))
                    
            pbar.set_postfix(rollout_error=rollout_error_mean, rollout_time_per_step=np.mean(rollout_times))
            
        with open(os.path.join(self.record_dir, "rollout_error_seed-{}.txt".format(self.load_seed)), "w") as f:
            json.dump(records, f)

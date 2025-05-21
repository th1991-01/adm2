import os
import gym
import copy
import d4rl
import neorl
import torch
import numpy as np
from tqdm import tqdm

import imageio
import matplotlib.pyplot as plt

from dynamics.mujoco_oracle_dynamics import MujocoOracleDynamics
from dynamics.adm_dynamics import ADMDynamics
from dynamics.sadm_dynamics import SADMDynamics
from components.static_fns import STATICFUNC
from env.model_as_sim import ADMSim, SADMSim
from agent.sac import SACAgent
from agent.td3 import TD3Agent
from agent.ppo import PPOAgent
from buffer.buffer4seqsamp import ReplayBufferForSeqSampling


class ModelSimEvaluator:
    """ model-as-simulator evaluator """

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

        if args.env == "neorl": args.env_name += f"-{args.data_type}"
        self.load_time = args.load_time
        self.load_seed = args.load_seed
        self.render_dir = f"./result/{args.env}/{args.env_name}/{args.load_label}/{self.load_time}/render"
        os.makedirs(self.render_dir, exist_ok=True)
        self.mujoco_img_dir = os.path.join(self.render_dir, "mujoco_imgs")
        self.model_img_dir = os.path.join(self.render_dir, "model_imgs")
        self.vs_dir = os.path.join(self.render_dir, "comparison")
        os.makedirs(self.mujoco_img_dir, exist_ok=True)
        os.makedirs(self.model_img_dir, exist_ok=True)
        os.makedirs(self.vs_dir, exist_ok=True)

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
        
        if args.algo == "sac":
            self.agent = SACAgent(
                obs_shape=args.obs_shape,
                hidden_dims=args.ac_hidden_dims,
                action_dim=args.action_dim,
                action_space=args.action_space,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                tau=args.tau,
                gamma=args.gamma,
                alpha=args.alpha,
                auto_alpha=args.auto_alpha,
                alpha_lr=args.alpha_lr,
                target_entropy=args.target_entropy,
                deterministic_backup=args.deterministic_backup,
                q_clip=args.q_clip,
                device=args.device
            )
        
        elif args.algo == "td3":
            self.agent = TD3Agent(
                obs_shape=args.obs_shape,
                hidden_dims=args.ac_hidden_dims,
                action_dim=args.action_dim,
                action_space=args.action_space,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                tau=args.tau,
                gamma=args.gamma,
                q_clip=args.q_clip,
                explore_noise=args.explore_noise,
                target_policy_noise=args.target_policy_noise,
                noise_clip=args.noise_clip,
                device=args.device
            )
            
        elif args.algo == "ppo":
            self.agent = PPOAgent(
                obs_shape=args.obs_shape,
                hidden_dims=args.ac_hidden_dims,
                action_dim=args.action_dim,
                action_space=args.action_space,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                clip_ratio=args.clip_ratio,
                value_clip=args.value_clip,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                ppo_epoch=args.ppo_epoch,
                mini_batch_size=args.batch_size,
                device=args.device
            )
            
        load_path = os.path.join(self.load_dir, "agent_seed-{}.pth".format(self.load_seed))
        self.agent.load_model(load_path)

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
        self.rollout_batch_size = 1
        self.rollout_length = args.rollout_length
        self.device = args.device
        self.seed = args.seed
        
    def run(self):
        # mujoco env
        self.eval_mujoco_env = MujocoOracleDynamics(self.env)
        
        # build model-based env
        extra_params = {"ood_terminate": False}
        eval_init_seqs = self.dataset.sample_all_head_nstep(self.n_starts-1)
        eval_init_seqs["s"] = torch.cat((eval_init_seqs["s"], eval_init_seqs["s_"][:, -1:]), dim=1)
        self.eval_model_env = self.ModelSim(
            dynamics=copy.deepcopy(self.dyna_model),
            static_fn=self.static_fn,
            max_steps=self.rollout_length,
            init_obs_seqs=eval_init_seqs["s"],
            init_act_seqs=eval_init_seqs["a"],
            n_parallels=self.rollout_batch_size,
            **extra_params
        )
        init_obs = self.eval_model_env.reset_all()
        
        # roll-out in mujoco env
        mujoco_imgs = []
        obs = copy.deepcopy(init_obs).flatten().cpu().numpy()
        for _ in tqdm(range(self.rollout_length), desc="Roll-out in MujocoEnv"):
            action = self.agent.act(obs, deterministic=True).cpu().numpy()
            obs, _, _, _ = self.eval_mujoco_env.step(obs, action)
            mujoco_imgs.append(self.eval_mujoco_env.env.render(mode="rgb_array"))
            
        for i in range(len(mujoco_imgs)):
            if i % 100 == 0:
                plt.figure()
                plt.imshow(mujoco_imgs[i])
                plt.axis("off")
                plt.savefig(os.path.join(self.mujoco_img_dir, f"step-{i}.pdf"), bbox_inches="tight", pad_inches=0)
                plt.savefig(os.path.join(self.mujoco_img_dir, f"step-{i}.png"), bbox_inches="tight", pad_inches=0)
                plt.close()
            
        # roll-out in model env
        model_imgs = []
        obs = copy.deepcopy(init_obs)
        for _ in tqdm(range(self.rollout_length), desc="Roll-out in ModelEnv"):
            action = self.agent.act(obs, deterministic=True)
            obs, _, _, _, _ = self.eval_model_env.step(action)
            self.eval_mujoco_env._set_state_from_obs(obs.flatten().cpu().numpy())
            model_imgs.append(self.eval_mujoco_env.env.render(mode="rgb_array"))
            
        for i in range(len(model_imgs)):
            if i % 100 == 0:
                plt.figure()
                plt.imshow(model_imgs[i])
                plt.axis("off")
                plt.savefig(os.path.join(self.model_img_dir, f"step-{i}.pdf"), bbox_inches="tight", pad_inches=0)
                plt.savefig(os.path.join(self.model_img_dir, f"step-{i}.png"), bbox_inches="tight", pad_inches=0)
                plt.close()
                
        # comparison mp4
        writer = imageio.get_writer(os.path.join(self.vs_dir, "comparison.mp4"), fps=60)
        for i in range(self.rollout_length):
            vs_img = np.concatenate((model_imgs[i], mujoco_imgs[i]), axis=1)
            writer.append_data(vs_img)
        writer.close()
            

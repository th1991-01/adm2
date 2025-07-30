import os
import copy
import json
import torch
import numpy as np
from tqdm import tqdm

from dynamics.adm_dynamics import ADMDynamics
from dynamics.sadm_dynamics import SADMDynamics
from dynamics.ensemble_dynamics import EnsembleDynamics
from dynamics.rnn_dynamics import RNNDynamics
from components.static_fns import STATICFUNC
from env.model_as_sim import ADMSim, SADMSim, EnSim, RNNSim
from agent.sac import SACAgent
from agent.td3 import TD3Agent
from agent.ppo import PPOAgent
from buffer.buffer import ReplayBuffer
from buffer.buffer4seqsamp import ReplayBufferForSeqSampling
from buffer.buffer4rollout import RolloutBuffer

from .base_trainer import BASETrainer

class ModelSimTrainer(BASETrainer):
    """ model-as-simulator trainer """

    def __init__(self, args):
        if args.env == "neorl":
            task, data_type, version = tuple(args.env_name.split('-'))
            args.env_name = task + '-' + version
            args.data_type = data_type

        super(ModelSimTrainer, self).__init__(args)

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
        
        self.on_policy = False
        self.off_policy = False
        
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
            self.off_policy = True
        
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
            self.off_policy = True
            
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
            self.on_policy = True
            self.ppo_slice = args.ppo_slice
            
        self.agent.train()
        self.penalty_coef = args.penalty_coef
        self.real_ratio = args.real_ratio

        # lr schedule
        if args.lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.agent.actor_optim, args.n_epochs)
        else:
            self.lr_scheduler = None

        # init replay buffer to store environmental data
        self.dataset = ReplayBufferForSeqSampling(
            buffer_size=1000000,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim,
            device=args.device,
        )
        rew_bias = 1 if args.env == "maze" else 0
        if args.env == "neorl":
            dataset, _ = self.env.get_dataset(data_type=args.data_type, train_num=1000, need_val=False)
            self.dataset.load_neorl_dataset(dataset, rew_bias)
        else:
            self.dataset.load_dataset(self.env.get_dataset(), rew_bias)

        # creat memory to store model-sim data
        if self.off_policy:
            self.model_memory = ReplayBuffer(
                buffer_size=args.buffer_size,
                obs_shape=args.obs_shape,
                action_dim=args.action_dim,
                device=args.device
            )
        elif self.on_policy:
            self.model_memory = RolloutBuffer(
                n_envs=args.rollout_batch_size,
                n_steps=args.ppo_slice,
                obs_shape=args.obs_shape,
                action_dim=args.action_dim,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                device=args.device
            )

        # other parameters
        self.model_lr = args.model_lr
        self.max_adm_step = args.max_adm_step
        self.n_starts = min(args.max_adm_step, args.n_starts)
        self.rollout_batch_size = args.rollout_batch_size
        self.rollout_length = args.rollout_length
        self.given_reward = args.given_reward
        self.warmup_steps = args.warmup_steps
        self.n_epochs = args.n_epochs
        self.step_per_epoch = args.step_per_epoch
        self.updates_per_step = args.updates_per_step
        
    def run(self):
        if self.load_model:
            # load dynamics model
            load_path = os.path.join(self.load_dir, "dyna_seed-{}.pth".format(self.load_seed))
            state_dict = torch.load(load_path)
            self.dyna_model.load_state_dict(state_dict)
        else:
            # learn dynamics model
            if self.args.dyna_model in ["sadm", "adm", "rnn", "dreamer"]:
                holdout_losses = self.dyna_model.learn_from(
                    max_adm_step=self.max_adm_step,
                    buffer=self.dataset,
                    lr=self.model_lr,
                    batch_size=1024
                )
            elif self.args.dyna_model == "en":
                holdout_losses = self.dyna_model.learn_from(
                    buffer=self.dataset,
                    lr=self.model_lr,
                    batch_size=1024
                )
            self._save({})
            with open(os.path.join(self.record_dir, "model_record_seed-{}.txt".format(self.seed)), "w") as f:
                json.dump({"model_loss": holdout_losses}, f)
        
        # build model-based env
        init_seqs = self.dataset.sample_all_nstep(self.n_starts-1)
        init_seqs["s"] = torch.cat((init_seqs["s"], init_seqs["s_"][:, -1:]), dim=1)
        self.model_env = self.ModelSim(
            dynamics=self.dyna_model,
            static_fn=self.static_fn,
            max_steps=self.rollout_length,
            init_obs_seqs=init_seqs["s"],
            init_act_seqs=init_seqs["a"],
            n_parallels=self.rollout_batch_size,
            given_reward=self.given_reward
        )
        
        eval_init_seqs = self.dataset.sample_all_head_nstep(self.n_starts-1)
        eval_init_seqs["s"] = torch.cat((eval_init_seqs["s"], eval_init_seqs["s_"][:, -1:]), dim=1)
        self.eval_model_env = self.ModelSim(
            dynamics=copy.deepcopy(self.dyna_model),
            static_fn=self.static_fn,
            max_steps=self.rollout_length,
            init_obs_seqs=eval_init_seqs["s"],
            init_act_seqs=eval_init_seqs["a"],
            n_parallels=self.eval_n_episodes,
            given_reward=self.given_reward
        )
        
        if self.off_policy:
            self._offpolicy_learn()
        elif self.on_policy:
            self._onpolicy_learn()

    def _offpolicy_learn(self):
        """ off-policy RL """

        # init
        records = {
            "epoch": [], "loss": {"actor": [], "critic1": [], "critic2": []}, "alpha": [], 
            "reward_mean": [], "reward_std": [], "reward_min": [], "reward_max": [],
            "reward_mean_in_model": [], "reward_std_in_model": [], "reward_min_in_model": [], "reward_max_in_model": [],
            "length_mean": [], "length_std": [], "length_min": [], "length_max": [],
            "length_mean_in_model": [], "length_std_in_model": [], "length_min_in_model": [], "length_max_in_model": [],
            "score_mean": [], "score_std": [], "score_min": [], "score_max": []
        }
        actor_loss, critic1_loss, critic2_loss, alpha = [None]*4
        eval_length_in_model, eval_length, eval_reward_in_model, eval_reward, eval_score = [None]*5
        obs = self.model_env.reset_all()
        num_steps = 0

        for e in range(self.n_epochs):
            
            pbar = tqdm(
                range(self.step_per_epoch), 
                desc="[Epoch {}] Training {} on ModelSim (task: {}.{}, seed: {})".format(
                    e, self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed)
            )
            for _ in pbar:
                # step
                action = self.agent.act(obs)
                next_obs, reward, uncertainty, terminated, truncated = self.model_env.step(action)
                reward -= self.penalty_coef * uncertainty
                terminal = terminated | truncated
                self.model_memory.store_batch(obs, action, reward, next_obs, terminated.float(), truncated.float())
                obs = next_obs
                num_steps += 1
                
                if terminal.any():
                    obs = self.model_env.reset(torch.where(terminal)[0])
                
                # update policy
                if num_steps >= self.warmup_steps:
                    for _ in range(self.updates_per_step):
                        real_sample_size = int(self.batch_size*self.real_ratio)
                        model_sample_size = self.batch_size - real_sample_size
                        real_batch = self.dataset.sample(batch_size=real_sample_size)
                        model_batch = self.model_memory.sample(batch_size=model_sample_size)
                        batch = {key: torch.cat(
                            (real_batch[key], model_batch[key]), axis=0) for key in real_batch.keys()}
                        batch.pop("timeout")
                        learning_info = self.agent.learn(**batch)
                        actor_loss = learning_info["loss"]["actor"]
                        critic1_loss = learning_info["loss"]["critic1"]
                        critic2_loss = learning_info["loss"]["critic2"]
                        alpha = learning_info.get("alpha", -1)

                pbar.set_postfix(
                    alpha=alpha,
                    actor_loss=actor_loss, 
                    critic1_loss=critic1_loss, 
                    critic2_loss=critic2_loss, 
                    eval_reward_in_model=eval_reward_in_model,
                    eval_reward=eval_reward,
                    eval_length_in_model=eval_length_in_model,
                    eval_length=eval_length,
                    eval_score=eval_score
                )

            # update lr
            if self.lr_scheduler is not None and num_steps >= self.warmup_steps:
                self.lr_scheduler.step()

            # evaluate policy
            episode_rewards, episode_lengths = self._eval_policy()
            episode_rewards_in_model, episode_lengths_in_model = self._eval_policy_in_model()
            records["epoch"].append(e)
            records["loss"]["actor"].append(actor_loss)
            records["loss"]["critic1"].append(critic1_loss)
            records["loss"]["critic2"].append(critic2_loss)
            records["alpha"].append(alpha)
            records["reward_mean"].append(float(np.mean(episode_rewards)))
            records["reward_std"].append(float(np.std(episode_rewards)))
            records["reward_min"].append(float(np.min(episode_rewards)))
            records["reward_max"].append(float(np.max(episode_rewards)))
            records["length_mean"].append(float(np.mean(episode_lengths)))
            records["length_std"].append(float(np.std(episode_lengths)))
            records["length_min"].append(float(np.min(episode_lengths)))
            records["length_max"].append(float(np.max(episode_lengths)))
            records["reward_mean_in_model"].append(float(np.mean(episode_rewards_in_model)))
            records["reward_std_in_model"].append(float(np.std(episode_rewards_in_model)))
            records["reward_min_in_model"].append(float(np.min(episode_rewards_in_model)))
            records["reward_max_in_model"].append(float(np.max(episode_rewards_in_model)))
            records["length_mean_in_model"].append(float(np.mean(episode_lengths_in_model)))
            records["length_std_in_model"].append(float(np.std(episode_lengths_in_model)))
            records["length_min_in_model"].append(float(np.min(episode_lengths_in_model)))
            records["length_max_in_model"].append(float(np.max(episode_lengths_in_model)))
            eval_reward = records["reward_mean"][-1]
            eval_reward_in_model = records["reward_mean_in_model"][-1]
            eval_length = records["length_mean"][-1]
            eval_length_in_model = records["length_mean_in_model"][-1]
            
            if actor_loss is not None:
                self.logger.add_scalar("loss/actor", actor_loss, e)
                self.logger.add_scalar("loss/critic1", critic1_loss, e)
                self.logger.add_scalar("loss/critic2", critic2_loss, e)
                self.logger.add_scalar("alpha", alpha, e)
            self.logger.add_scalar("eval/reward", eval_reward, e)
            self.logger.add_scalar("eval/reward_in_model", eval_reward_in_model, e)
            self.logger.add_scalar("eval/length", eval_length, e)
            self.logger.add_scalar("eval/length_in_model", eval_length_in_model, e)

            records["score_mean"].append(self.score_func(records["reward_mean"][-1])*100)
            records["score_std"].append(self.score_func(records["reward_std"][-1])*100)
            records["score_min"].append(self.score_func(records["reward_min"][-1])*100)
            records["score_max"].append(self.score_func(records["reward_max"][-1])*100)
            eval_score = self.score_func(eval_reward)*100
            self.logger.add_scalar("eval/score", eval_score, e)

            # save
            self._save(records)

        self.logger.close()
        
    def _onpolicy_learn(self):
        """ on-policy RL """

        # init
        records = {
            "epoch": [], "loss": {"actor": [], "critic": []}, "kl": [], "value": [],
            "reward_mean": [], "reward_std": [], "reward_min": [], "reward_max": [],
            "reward_mean_in_model": [], "reward_std_in_model": [], "reward_min_in_model": [], "reward_max_in_model": [],
            "length_mean": [], "length_std": [], "length_min": [], "length_max": [],
            "length_mean_in_model": [], "length_std_in_model": [], "length_min_in_model": [], "length_max_in_model": [],
            "score_mean": [], "score_std": [], "score_min": [], "score_max": []
        }
        actor_loss, critic_loss, kl, value = [None]*4
        eval_length_in_model, eval_length, eval_reward_in_model, eval_reward, eval_score = [None]*5
        obs = self.model_env.reset_all()
        num_steps = 0

        for e in range(self.n_epochs):
            
            pbar = tqdm(
                range(self.step_per_epoch), 
                desc="[Epoch {}] Training {} on ModelSim (task: {}.{}, seed: {})".format(
                    e, self.args.algo.upper(), self.args.env.upper(), self.args.env_name, self.seed)
            )
            for _ in pbar:
                # step
                action, log_prob, value = self.agent.act_and_value(obs)
                next_obs, reward, uncertainty, terminated, truncated = self.model_env.step(action)
                reward -= self.penalty_coef * uncertainty
                terminal = terminated | truncated
                self.model_memory.store_batch(obs, action, reward, value, log_prob)
                obs = next_obs
                num_steps += 1
                
                if terminal.any() or num_steps % self.ppo_slice == 0:
                    _, _, value = self.agent.actor4ward(obs)
                    value[terminated] = 0
                    if num_steps % self.ppo_slice == 0:
                        self.model_memory.finish_episode(torch.arange(self.model_env.n_parallels), value)
                    else:
                        self.model_memory.finish_episode(torch.where(terminal)[0], value[terminal])
                    
                    if terminal.any():
                        obs = self.model_env.reset(torch.where(terminal)[0])
                
                # update policy
                if num_steps % self.ppo_slice == 0:
                    batch = self.model_memory.sample_all()
                    learning_info = self.agent.learn(**batch)
                    actor_loss = learning_info["loss"]["actor"]
                    critic_loss = learning_info["loss"]["critic"]
                    kl = learning_info.get("kl", -1)
                    value = learning_info.get("value", -1)

                pbar.set_postfix(
                    kl=kl,
                    actor_loss=actor_loss, 
                    critic_loss=critic_loss,
                    eval_reward_in_model=eval_reward_in_model,
                    eval_reward=eval_reward,
                    eval_length_in_model=eval_length_in_model,
                    eval_length=eval_length,
                    eval_score=eval_score
                )

            # update lr
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # evaluate policy
            episode_rewards, episode_lengths = self._eval_policy()
            episode_rewards_in_model, episode_lengths_in_model = self._eval_policy_in_model()
            records["epoch"].append(e)
            records["loss"]["actor"].append(actor_loss)
            records["loss"]["critic"].append(critic_loss)
            records["kl"].append(kl)
            records["value"].append(value)
            records["reward_mean"].append(float(np.mean(episode_rewards)))
            records["reward_std"].append(float(np.std(episode_rewards)))
            records["reward_min"].append(float(np.min(episode_rewards)))
            records["reward_max"].append(float(np.max(episode_rewards)))
            records["length_mean"].append(float(np.mean(episode_lengths)))
            records["length_std"].append(float(np.std(episode_lengths)))
            records["length_min"].append(float(np.min(episode_lengths)))
            records["length_max"].append(float(np.max(episode_lengths)))
            records["reward_mean_in_model"].append(float(np.mean(episode_rewards_in_model)))
            records["reward_std_in_model"].append(float(np.std(episode_rewards_in_model)))
            records["reward_min_in_model"].append(float(np.min(episode_rewards_in_model)))
            records["reward_max_in_model"].append(float(np.max(episode_rewards_in_model)))
            records["length_mean_in_model"].append(float(np.mean(episode_lengths_in_model)))
            records["length_std_in_model"].append(float(np.std(episode_lengths_in_model)))
            records["length_min_in_model"].append(float(np.min(episode_lengths_in_model)))
            records["length_max_in_model"].append(float(np.max(episode_lengths_in_model)))
            eval_reward = records["reward_mean"][-1]
            eval_length = records["length_mean"][-1]
            eval_reward_in_model = records["reward_mean_in_model"][-1]
            eval_length_in_model = records["length_mean_in_model"][-1]
            
            if actor_loss is not None:
                self.logger.add_scalar("loss/actor", actor_loss, e)
                self.logger.add_scalar("loss/critic", critic_loss, e)
                self.logger.add_scalar("kl", kl, e)
                self.logger.add_scalar("value", value, e)
            self.logger.add_scalar("eval/reward", eval_reward, e)
            self.logger.add_scalar("eval/length", eval_length, e)
            self.logger.add_scalar("eval/reward_in_model", eval_reward_in_model, e)
            self.logger.add_scalar("eval/length_in_model", eval_length_in_model, e)

            records["score_mean"].append(self.score_func(records["reward_mean"][-1])*100)
            records["score_std"].append(self.score_func(records["reward_std"][-1])*100)
            records["score_min"].append(self.score_func(records["reward_min"][-1])*100)
            records["score_max"].append(self.score_func(records["reward_max"][-1])*100)
            eval_score = self.score_func(eval_reward)*100
            self.logger.add_scalar("eval/score", eval_score, e)

            # save
            self._save(records)

        self.logger.close()
        
    def _eval_policy_in_model(self):
        episode_rewards = torch.zeros(self.eval_n_episodes, dtype=torch.float32, device=self.device)
        episode_lengths = torch.zeros(self.eval_n_episodes, dtype=torch.float32, device=self.device)
        done = torch.zeros(self.eval_n_episodes, dtype=torch.bool, device=self.device)
        obs = self.eval_model_env.reset_all()
        while not done.all():
            action = self.agent.act(obs, deterministic=True)
            obs, reward, _, terminated, truncated = self.eval_model_env.step(action)
            episode_rewards[~done] += reward.flatten()[~done]
            episode_lengths[~done] += 1
            done[~done] = (terminated | truncated).flatten()[~done]
        return episode_rewards.cpu().numpy().tolist(), episode_lengths.cpu().numpy().tolist()

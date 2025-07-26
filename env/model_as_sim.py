import torch
import numpy as np

import gym


class ADMSim(gym.Env):
    
    def __init__(
        self,
        dynamics,
        static_fn,
        max_steps,
        init_obs_seqs,
        init_act_seqs,
        n_parallels,
        given_reward
    ):
        self.dynamics = dynamics
        self.static_fn = static_fn
        self.max_steps = max_steps
        self.init_obs_seqs = init_obs_seqs                  # m steps
        self.init_act_seqs = init_act_seqs                  # m-1 steps
        self.m = self.init_obs_seqs.shape[1]
        self.n_choices = self.init_obs_seqs.shape[0]
        self.n_parallels = n_parallels
        self.given_reward = given_reward

    @ torch.no_grad()
    def reset_all(self):
        sample_ids = np.random.randint(0, self.n_choices, self.n_parallels)
        self._obs_seq = self.init_obs_seqs[sample_ids]
        self._act_seq = self.init_act_seqs[sample_ids]
        self._cnt = torch.zeros((self.n_parallels, 1), device=self._obs_seq.device)
        return self._obs_seq[:, -1]
        
    @ torch.no_grad()
    def reset(self, env_ids):
        sample_ids = np.random.randint(0, self.n_choices, len(env_ids))
        self._obs_seq[env_ids] = self.init_obs_seqs[sample_ids]
        self._act_seq[env_ids] = self.init_act_seqs[sample_ids]
        self._cnt[env_ids] = 0
        return self._obs_seq[:, -1]
        
    @ torch.no_grad()
    def step(self, action):
        self._act_seq = torch.cat((self._act_seq, action[:, None]), dim=1)
        
        next_obs_means = []
        next_obs_stds = []
        reward_means = []
        reward_stds = []
        for k in range(1, self.m+1):
            input_obs = self._obs_seq[:, -k]
            input_act = self._act_seq[:, -k:]
            next_obs_mean, next_obs_std, reward_mean, reward_std = \
                self.dynamics.dyna_dist(input_obs, input_act)
            next_obs_means.append(next_obs_mean)
            next_obs_stds.append(next_obs_std)
            reward_means.append(reward_mean)
            reward_stds.append(reward_std)
            
        # random choice
        k = np.random.randint(self.m)
        next_obs = torch.normal(next_obs_means[k], next_obs_stds[k])
        if self.given_reward:
            reward = self.static_fn.reward_fn(
                self._obs_seq[:, -1].detach().cpu().numpy(),
                self._act_seq[:, -1].detach().cpu().numpy(),
                next_obs.detach().cpu().numpy()
            )
            reward = torch.as_tensor(reward, dtype=torch.float32, device=next_obs.device)
        else:
            reward = torch.normal(reward_means[k], reward_stds[k])
        
        # uncertainty
        next_obs_means = torch.stack(next_obs_means, dim=0)
        uncertainty = torch.sqrt(next_obs_means.var(dim=0).mean(dim=-1, keepdim=True))
        
        self._cnt += 1
        terminated = self.static_fn.termination_fn(
            self._obs_seq[:, -1].detach().cpu().numpy(),
            self._act_seq[:, -1].detach().cpu().numpy(),
            next_obs.detach().cpu().numpy()
        )
        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=next_obs.device)
        truncated = self._cnt >= self.max_steps
        
        self._obs_seq = torch.cat((self._obs_seq[:, 1:], next_obs[:, None]), dim=1)
        self._act_seq = self._act_seq[:, 1:]
        return next_obs, reward, uncertainty, terminated, truncated
        
        
class SADMSim(ADMSim):
    
    def __init__(
        self,
        dynamics,
        static_fn,
        max_steps,
        init_obs_seqs,
        init_act_seqs,
        n_parallels,
        given_reward
    ):
        super().__init__(
            dynamics, static_fn, max_steps,
            init_obs_seqs, init_act_seqs, n_parallels, given_reward
        )
        self.max_adm_step = self.dynamics.max_adm_step
        
    @ torch.no_grad()
    def reset_all(self):
        sample_ids = np.random.randint(0, self.n_choices, self.n_parallels)
        self._obs_seq = self.init_obs_seqs[sample_ids]
        self._act_seq = self.init_act_seqs[sample_ids]
        init_hiddens = self.dynamics.init_hiddens(self._obs_seq, self._act_seq)
        self.dynamics.set_hiddens(init_hiddens)
        self._cnt = torch.zeros((self.n_parallels, 1), device=self._obs_seq.device)
        return self._obs_seq[:, -1]
        
    @ torch.no_grad()
    def reset(self, env_ids):
        sample_ids = np.random.randint(0, self.n_choices, len(env_ids))
        self._obs_seq[env_ids] = self.init_obs_seqs[sample_ids]
        self._act_seq[env_ids] = self.init_act_seqs[sample_ids]
        init_hiddens = self.dynamics.init_hiddens(self._obs_seq[env_ids], self._act_seq[env_ids])
        self.dynamics.set_hiddens(init_hiddens, env_ids)
        self._cnt[env_ids] = 0
        return self._obs_seq[:, -1]
        
    @ torch.no_grad()
    def step(self, action):
        self._act_seq = torch.cat((self._act_seq, action[:, None]), dim=1)
        
        next_obs_means, next_obs_stds, reward_means, reward_stds = \
            self.dynamics.dyna_dist(self._obs_seq[:, -1], self._act_seq[:, -1])
            
        # random choice
        k = np.random.randint(self.m)
        next_obs = torch.normal(next_obs_means[k], next_obs_stds[k])
        if self.given_reward:
            reward = self.static_fn.reward_fn(
                self._obs_seq[:, -1].detach().cpu().numpy(),
                self._act_seq[:, -1].detach().cpu().numpy(),
                next_obs.detach().cpu().numpy()
            )
            reward = torch.as_tensor(reward, dtype=torch.float32, device=action.device)
        else:
            reward = torch.normal(reward_means[k], reward_stds[k])
        
        # uncertainty
        uncertainty = torch.sqrt(next_obs_means.var(dim=0).mean(dim=-1, keepdim=True))
        
        self._cnt += 1
        terminated = self.static_fn.termination_fn(
            self._obs_seq[:, -1].detach().cpu().numpy(),
            self._act_seq[:, -1].detach().cpu().numpy(),
            next_obs.detach().cpu().numpy()
        )
        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=next_obs.device)
        truncated = self._cnt >= self.max_steps
        
        residual = (self._cnt + self.m - 1) % self.max_adm_step
        h_update_ids = torch.where((residual >= 0) & (residual < self.m))[0]
        if h_update_ids.any():
            new_hs = self.dynamics.encode_obs(next_obs[h_update_ids])
            self.dynamics.update_hiddens(new_hs, h_update_ids)
        
        self._obs_seq = torch.cat((self._obs_seq[:, 1:], next_obs[:, None]), dim=1)
        self._act_seq = self._act_seq[:, 1:]
        return next_obs, reward, uncertainty, terminated, truncated


class EnSim(gym.Env):
    
    def __init__(
        self,
        dynamics,
        static_fn,
        max_steps,
        init_obs_seqs,
        init_act_seqs,
        n_parallels,
        given_reward
    ):
        self.dynamics = dynamics
        self.static_fn = static_fn
        self.max_steps = max_steps
        self.init_obs_seqs = init_obs_seqs                  # m steps
        self.init_act_seqs = init_act_seqs                  # m-1 steps
        self.n_choices = self.init_obs_seqs.shape[0]
        self.n_parallels = n_parallels
        self.given_reward = given_reward
        
    @ torch.no_grad()
    def reset_all(self):
        sample_ids = np.random.randint(0, self.n_choices, self.n_parallels)
        self._obs_seq = self.init_obs_seqs[sample_ids]
        self._act_seq = self.init_act_seqs[sample_ids]
        self._cnt = torch.zeros((self.n_parallels, 1), device=self._obs_seq.device)
        return self._obs_seq[:, -1]
        
    @ torch.no_grad()
    def reset(self, env_ids):
        sample_ids = np.random.randint(0, self.n_choices, len(env_ids))
        self._obs_seq[env_ids] = self.init_obs_seqs[sample_ids]
        self._act_seq[env_ids] = self.init_act_seqs[sample_ids]
        self._cnt[env_ids] = 0
        return self._obs_seq[:, -1]
        
    @ torch.no_grad()
    def step(self, action):
        self._act_seq = torch.cat((self._act_seq, action[:, None]), dim=1)
        
        next_obs_means, next_obs_stds, reward_means, reward_stds = \
            self.dynamics.dyna_dist(self._obs_seq[:, -1], self._act_seq[:, -1])
            
        # random choice
        k = np.random.randint(self.dynamics.model.num_elites)
        next_obs = torch.normal(next_obs_means[k], next_obs_stds[k].clamp(0))
        if self.given_reward:
            reward = self.static_fn.reward_fn(
                self._obs_seq[:, -1].detach().cpu().numpy(),
                self._act_seq[:, -1].detach().cpu().numpy(),
                next_obs.detach().cpu().numpy()
            )
            reward = torch.as_tensor(reward, dtype=torch.float32, device=action.device)
        else:
            reward = torch.normal(reward_means[k], reward_stds[k])
        
        # uncertainty
        uncertainty = torch.sqrt(next_obs_means.var(dim=0).mean(dim=-1, keepdim=True))
        
        self._cnt += 1
        terminated = self.static_fn.termination_fn(
            self._obs_seq[:, -1].detach().cpu().numpy(),
            self._act_seq[:, -1].detach().cpu().numpy(),
            next_obs.detach().cpu().numpy()
        )
        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=next_obs.device)
        truncated = self._cnt >= self.max_steps
        
        self._obs_seq = torch.cat((self._obs_seq[:, 1:], next_obs[:, None]), dim=1)
        self._act_seq = self._act_seq[:, 1:]
        return next_obs, reward, uncertainty, terminated, truncated

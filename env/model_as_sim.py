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
        n_parallels
    ):
        self.dynamics = dynamics
        self.static_fn = static_fn
        self.max_steps = max_steps
        self.init_obs_seqs = init_obs_seqs                  # m steps
        self.init_act_seqs = init_act_seqs                  # m-1 steps
        self.m = self.init_obs_seqs.shape[1]
        self.n_choices = self.init_obs_seqs.shape[0]
        self.n_parallels = n_parallels

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
        for k in range(1, self.m+1):
            input_obs = self._obs_seq[:, -k]
            input_act = self._act_seq[:, -k:]
            next_obs_mean, next_obs_std, _, _ = \
                self.dynamics.dyna_dist(input_obs, input_act)
            next_obs_means.append(next_obs_mean)
            next_obs_stds.append(next_obs_std)
            
        # random choice
        k = np.random.randint(self.m)
        next_obs = torch.normal(next_obs_means[k], next_obs_stds[k])
        reward = self.static_fn.reward_fn(
            self._obs_seq[:, -1].detach().cpu().numpy(),
            self._act_seq[:, -1].detach().cpu().numpy(),
            next_obs.detach().cpu().numpy()
        )
        reward = torch.as_tensor(reward, dtype=torch.float32, device=next_obs.device)
        
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
        ood_terminate=False,
        dev_thresh=1.0
    ):
        super().__init__(
            dynamics, static_fn, max_steps,
            init_obs_seqs, init_act_seqs, n_parallels
        )
        self.max_adm_step = self.dynamics.max_adm_step
        self.ood_terminate = ood_terminate
        self.dev_thresh = dev_thresh
        
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
        
        next_obs_means, next_obs_stds, _, _ = \
            self.dynamics.dyna_dist(self._obs_seq[:, -1], self._act_seq[:, -1])
            
        # random choice
        k = np.random.randint(self.m)
        next_obs = torch.normal(next_obs_means[k], next_obs_stds[k])
        reward = self.static_fn.reward_fn(
            self._obs_seq[:, -1].detach().cpu().numpy(),
            self._act_seq[:, -1].detach().cpu().numpy(),
            next_obs.detach().cpu().numpy()
        )
        reward = torch.as_tensor(reward, dtype=torch.float32, device=action.device)
        
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
        
        # fix terminated
        if self.ood_terminate:
            ood_terminated = (next_obs > self.dynamics.obs_max[None]) | (next_obs < self.dynamics.obs_min[None])
            next_h = self.dynamics.encode_obs(next_obs)
            next_obs_recon = self.dynamics.decode_h(next_h)
            ood_terminated = torch.pow(next_obs_recon-next_obs, 2) > self.dev_thresh
            ood_terminated = ood_terminated.any(dim=-1)
            terminated[ood_terminated] = True
        
        residual = (self._cnt + self.m - 1) % self.max_adm_step
        h_update_ids = torch.where((residual >= 0) & (residual < self.m))[0]
        if h_update_ids.any():
            new_hs = self.dynamics.encode_obs(next_obs[h_update_ids])
            self.dynamics.update_hiddens(new_hs, h_update_ids)
        
        self._obs_seq = torch.cat((self._obs_seq[:, 1:], next_obs[:, None]), dim=1)
        self._act_seq = self._act_seq[:, 1:]
        return next_obs, reward, uncertainty, terminated, truncated

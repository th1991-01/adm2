import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from .sadm import SADModel

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while maintaining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class SADMDynamics(nn.Module):
    """ Self-transition Any-step Dynamics """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=200,
        rnn_num_layers=3,
        max_adm_step=None,
        dropout=0,
        device="cuda:0"
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_adm_step = max_adm_step
        self.device = device

        self.model = SADModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rnn_num_layers=rnn_num_layers,
            dropout=dropout,
            device=device
        )

        # 'mean' and 'std' for normalization
        self.register_parameter("obs_mu", nn.Parameter(torch.zeros(self.obs_dim), requires_grad=False))
        self.register_parameter("obs_std", nn.Parameter(torch.zeros(self.obs_dim), requires_grad=False))
        self.register_parameter("act_mu", nn.Parameter(torch.zeros(self.action_dim), requires_grad=False))
        self.register_parameter("act_std", nn.Parameter(torch.zeros(self.action_dim), requires_grad=False))

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(self.obs_dim + 1) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(self.obs_dim + 1) * -10, requires_grad=True)
        )
        
        self.obs_max = None
        self.obs_min = None
        
        self.to(self.device)

    def set_mu_std(self, obs_mu, obs_std, act_mu, act_std):
        self.obs_mu.data = torch.as_tensor(obs_mu, dtype=torch.float32, device=self.device)
        self.obs_std.data = torch.as_tensor(obs_std, dtype=torch.float32, device=self.device)
        self.act_mu.data = torch.as_tensor(act_mu, dtype=torch.float32, device=self.device)
        self.act_std.data = torch.as_tensor(act_std, dtype=torch.float32, device=self.device)
        
    def set_max_min(self, obs_max, obs_min):
        self.obs_max = torch.as_tensor(obs_max, dtype=torch.float32, device=self.device)
        self.obs_min = torch.as_tensor(obs_min, dtype=torch.float32, device=self.device)
        
    @ torch.no_grad()
    def delta_obs_and_h(self, obs, action):
        # shape@obs: (bs, obs_dim)
        # shape@actions: (bs, h_step, act_dim)
        # normalization
        _obs = (obs - self.obs_mu) / self.obs_std
        _action = (action - self.act_mu) / self.act_std

        model_out, h = self.model(_obs, _action)
        mean, _ = torch.chunk(model_out, 2, dim=-1)
        return mean[:, :-1], h

    def forward(self, obs, action):
        # shape@obs: (bs, obs_dim)
        # shape@actions: (bs, h_step, act_dim)
        # normalization
        _obs = (obs - self.obs_mu) / self.obs_std
        _action = (action - self.act_mu) / self.act_std

        model_out, _ = self.model(_obs, _action)
        mean, logvar = torch.chunk(model_out, 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar
    
    def encode_obs(self, obs):
        obs = (obs - self.obs_mu) / self.obs_std
        return self.model.encode_obs(obs)
    
    def init_hiddens(self, obs_seq, act_seq):
        # obs_seq: (bs, m, -1)
        # act_seq: (bs, m-1, -1)
        _obs_seq = (obs_seq - self.obs_mu) / self.obs_std
        _act_seq = (act_seq - self.act_mu) / self.act_std
        return self.model.init_hiddens(_obs_seq, _act_seq)
    
    def set_hiddens(self, hiddens, env_ids=None):
        self.model.set_hiddens(hiddens, env_ids)
        
    def update_hiddens(self, hiddens, env_ids):
        self.model.update_hiddens(hiddens, env_ids)
        
    def transition_forward(self, action):
        # action: (bs, -1)
        _action = (action - self.act_mu) / self.act_std
        model_out = self.model.transition(_action)
        mean, logvar = torch.chunk(model_out, 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar
    
    @ torch.no_grad()
    def dyna_dist(self, obs, action):
        # obs: (bs, -1)
        # action: (bs, -1)
        mean, logvar = self.transition_forward(action)
        mean[..., :-1] += obs[None]
        std = torch.sqrt(torch.exp(logvar))
        next_obs_mean = mean[..., :-1]
        next_obs_std = std[..., :-1]
        reward_mean = mean[..., -1:]
        reward_std = std[..., -1:]
        return next_obs_mean, next_obs_std, reward_mean, reward_std
    
    def learn_from(self, max_adm_step, buffer, lr, batch_size, max_holdout=1000, min_epochs=1):
        """ learn any-step dynamics model """
        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        
        # set mean and std
        obs_mu, obs_std, act_mu, act_std = buffer.cal_mu_std()
        self.set_mu_std(obs_mu, obs_std, act_mu, act_std)
        saved_state_dict = copy.deepcopy(self.state_dict())

        data_size = buffer.size
        holdout_size = min(int(data_size * 0.2), max_holdout)
        train_size = data_size - holdout_size

        epoch = 0
        holdout_losses = [1e10] * max_adm_step
        cnt = 0

        while True:
            epoch += 1

            pbar = tqdm(range(train_size//batch_size), desc=f"[M][Epoch {epoch} @ Any-step Dynamics Model Training]")
            for _ in pbar:
                # sample any-step data
                k = np.random.randint(max_adm_step) + 1
                any_step_seq = buffer.sample_nstep(batch_size, k, end_idx=train_size)
                s = any_step_seq["s"][:, 0]
                a_seq = any_step_seq["a"]
                r = any_step_seq["r"][:, -1]
                s_ = any_step_seq["s_"][:, -1]
                trgt = torch.concatenate((s_-any_step_seq["s"][:, -1], r), dim=-1)

                # any-step loss
                mean, logvar = self.forward(s, a_seq)
                inv_var = torch.exp(-logvar)
                mse_loss = (torch.pow(mean - trgt, 2) * inv_var).mean()
                var_loss = logvar.mean()
                loss = mse_loss + var_loss
                # loss = loss + 0.01 * self.dynamics.max_logvar.sum() - 0.01 * self.dynamics.min_logvar.sum()
                
                # backward
                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.set_postfix(
                    train_loss=loss.item(),
                    holdout_loss=np.mean(holdout_losses),
                )

            new_val_losses, improve_ks = [], []
            s_list = []
            for k in range(1, max_adm_step+1):
                k_step_seq = buffer.sample_all_nstep(k, start_idx=train_size)
                s_list.append(k_step_seq["s_"][:, -1])
                k_val_loss = self.validate_from(
                    s=k_step_seq["s"],
                    a=k_step_seq["a"],
                    r=k_step_seq["r"][:, -1],
                    s_=k_step_seq["s_"][:, -1]
                )
                new_val_losses.append(k_val_loss)
                k_improvement = (holdout_losses[k-1] - k_val_loss) / holdout_losses[k-1]
                if k_improvement > 0:
                    improve_ks.append(k)

            if len(improve_ks) > 0 and np.mean(new_val_losses) < np.mean(holdout_losses):
                saved_state_dict = copy.deepcopy(self.state_dict())
                holdout_losses = new_val_losses
                cnt = 0
            else:
                cnt += 1

            if cnt >= 25 and epoch >= min_epochs:
                break

        self.load_state_dict(saved_state_dict)
        return holdout_losses
    
    @ torch.no_grad()
    def validate_from(self, s, a, r, s_):
        """ validate any-step dynamics model (fixed k-step validation) """
        trgt = torch.cat((s_ - s[:, -1], r), dim=-1)
        mean, _ = self.forward(s[:, 0], a)
        loss = ((mean - trgt) ** 2).mean()
        return float(loss.cpu().detach().numpy())
    
    def save_model(self, filepath):
        """ save model """
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """ load model """
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)

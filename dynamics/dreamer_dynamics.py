import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from .rnn import RNNModel, ResBlock, Swish

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while maintaining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class RSSM(nn.Module):
    """ Recurrent State-Space Model """
    def __init__(self, obs_dim, latent_dim, action_dim, hidden_dim=200, rnn_num_layers=3, dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: obs -> latent representation
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )
        
        # Deterministic state (RNN hidden state)
        self.rnn = nn.GRU(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True
        )
        
        # Stochastic state (latent)
        self.prior_net = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )
        
        # Posterior net (for inference)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )
        
        # Reward predictor
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # Decoder: (rnn_out + latent) -> obs (same input as reward_net)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, obs_dim*2)
        )
    
    def forward(self, obs, action, h_state=None):
        # obs: (bs, seq_len+1, obs_dim)
        # action: (bs, seq_len, action_dim)
        # h_state: (num_layers, bs, hidden_dim)
        
        batch_size, seq_len, _ = obs.shape
        
        # Encode observations to latent space
        obs_mean, obs_logvar = torch.chunk(self.encoder(obs), 2, dim=-1)  # (bs, seq_len, latent_dim)
        
        # Sample from posterior
        obs_std = torch.sqrt(torch.exp(obs_logvar))
        obs_sample = torch.normal(obs_mean, obs_std)
        
        # RSSM forward pass
        rnn_input = torch.cat([obs_sample[:, :-1], action], dim=-1)  # (bs, seq_len, latent_dim + action_dim)
        rnn_out, h_state = self.rnn(rnn_input, h_state)  # (bs, seq_len, hidden_dim)
        hs = torch.cat((torch.zeros_like(rnn_out[:, :1]), rnn_out), dim=1)
        
        # Prior distribution
        prior_out = self.prior_net(hs)  # (bs, seq_len+1, latent_dim*2)
        prior_mean, prior_logvar = torch.chunk(prior_out, 2, dim=-1)
        
        # Posterior distribution
        posterior_input = torch.cat([hs, obs_sample], dim=-1)  # (bs, seq_len+1, hidden_dim + latent_dim)
        posterior_out = self.posterior_net(posterior_input)  # (bs, seq_len+1, latent_dim*2)
        posterior_mean, posterior_logvar = torch.chunk(posterior_out, 2, dim=-1)
        
        # Sample from posterior for next step
        posterior_std = torch.sqrt(torch.exp(posterior_logvar))
        posterior_sample = torch.normal(posterior_mean, posterior_std)
        
        # Common input for both reward and decoder
        common_input = torch.cat([hs, posterior_sample], dim=-1)  # (bs, seq_len+1, hidden_dim + latent_dim)
        
        # Predict reward
        reward_mean, reward_logvar = torch.chunk(self.reward_net(common_input), 2, dim=-1)  # (bs, seq_len+1, 1)
        
        # Decode to obs space
        decoded_obs_mean, decoded_obs_logvar = torch.chunk(self.decoder(common_input), 2, dim=-1)  # (bs, seq_len+1, obs_dim)
        
        # Combine decoded_obs and reward for output
        output = torch.cat([decoded_obs_mean, reward_mean, decoded_obs_logvar, reward_logvar], dim=-1)  # (bs, seq_len+1, obs_dim + 1)
        
        return output, rnn_out, prior_mean, prior_logvar, posterior_mean, posterior_logvar

class DreamerDynamics(nn.Module):
    """ Dreamer Dynamics with RSSM """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=200,
        rnn_num_layers=3,
        max_adm_step=None,
        dropout=0.1,
        device="cuda:0",
        latent_dim=10
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.output_dim = (self.obs_dim + 1) * 2  # Keep original output format for compatibility
        self.max_adm_step = max_adm_step
        self.device = device

        # RSSM (includes encoder and decoder)
        self.rssm = RSSM(obs_dim, latent_dim, action_dim, hidden_dim, rnn_num_layers, dropout)

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

        self.to(self.device)

    def set_mu_std(self, obs_mu, obs_std, act_mu, act_std):
        self.obs_mu.data = torch.as_tensor(obs_mu, dtype=torch.float32, device=self.device)
        self.obs_std.data = torch.as_tensor(obs_std, dtype=torch.float32, device=self.device)
        self.act_mu.data = torch.as_tensor(act_mu, dtype=torch.float32, device=self.device)
        self.act_std.data = torch.as_tensor(act_std, dtype=torch.float32, device=self.device)

    def forward(self, obs, action):
        # shape@obs: (bs, h_step, obs_dim)
        # shape@actions: (bs, h_step, act_dim)
        # normalization
        _obs = (obs - self.obs_mu) / self.obs_std
        _action = (action - self.act_mu) / self.act_std

        # Encode observations to latent space
        obs_mean, obs_logvar = torch.chunk(self.rssm.encoder(_obs), 2, dim=-1)  # (bs, seq_len, latent_dim)
        
        # Sample from posterior
        obs_std = torch.sqrt(torch.exp(obs_logvar))
        obs_sample = torch.normal(obs_mean, obs_std)
        
        # RSSM forward pass
        rnn_input = torch.cat([obs_sample, _action], dim=-1)  # (bs, seq_len, latent_dim + action_dim)
        rnn_out, _ = self.rssm.rnn(rnn_input)  # (bs, seq_len, hidden_dim)
        
        # Prior distribution
        prior_out = self.rssm.prior_net(rnn_out)  # (bs, seq_len, latent_dim*2)
        prior_mean, prior_logvar = torch.chunk(prior_out, 2, dim=-1)
        
        prior_std = torch.sqrt(torch.exp(prior_logvar))
        prior_sample = torch.normal(prior_mean, prior_std)
        
        common_input = torch.cat([rnn_out, prior_sample], dim=-1)
        
        # Extract components from RSSM output
        decoded_obs_mean, decoded_obs_logvar = torch.chunk(self.rssm.decoder(common_input), 2, dim=-1)
        reward_mean, reward_logvar = torch.chunk(self.rssm.reward_net(common_input), 2, dim=-1)
        
        # Combine decoded_obs and reward for output
        mean = torch.cat([decoded_obs_mean, reward_mean], dim=-1)
        logvar = torch.cat([decoded_obs_logvar, reward_logvar], dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        
        return mean[:, -1], logvar[:, -1]
    
    @ torch.no_grad()
    def dyna_dist(self, obs, action):
        mean, logvar = self.forward(obs, action)
        std = torch.sqrt(torch.exp(logvar))
        next_obs_mean = mean[:, :-1]
        next_obs_std = std[:, :-1]
        reward_mean = mean[:, -1:]
        reward_std = std[:, -1:]
        return next_obs_mean, next_obs_std, reward_mean, reward_std

    @ torch.no_grad()
    def step(self, obs, action):
        mean, logvar = self.forward(obs, action)
        std = torch.sqrt(torch.exp(logvar))
        sample = torch.normal(mean, std)
        next_obs = sample[:, :-1]
        reward = sample[:, -1:]
        return next_obs, reward

    @ torch.no_grad()
    def dstep(self, obs, action):
        """ deterministic step """
        mean, _ = self.forward(obs, action)
        return mean[:, :-1], mean[:, -1:]
    
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

            pbar = tqdm(range(train_size//batch_size), desc=f"[M][Epoch {epoch} @ Dreamer Dynamics Model Training]")
            for _ in pbar:
                # sample any-step data
                k = np.random.randint(max_adm_step) + 1
                any_step_seq = buffer.sample_nstep(batch_size, k+1, end_idx=train_size)
                s = any_step_seq["s"]
                a_seq = any_step_seq["a"][:, :-1]
                r = any_step_seq["r"]
                r = torch.cat((torch.zeros_like(r[:, :1]), r[:, :-1]), dim=1)

                # Normalize inputs
                _s = (s - self.obs_mu) / self.obs_std
                _a_seq = (a_seq - self.act_mu) / self.act_std

                # Dreamer loss components
                rssm_out, _, prior_mean, prior_logvar, posterior_mean, posterior_logvar = self.rssm(_s, _a_seq)
                
                # Extract components from RSSM output
                mean, logvar = torch.chunk(rssm_out, 2, dim=-1)
                logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
                decoded_obs_mean = mean[:, :, :self.obs_dim]  # (bs, obs_dim)
                decoded_obs_logvar = logvar[:, :, :self.obs_dim]
                predicted_reward_mean = mean[:, :, self.obs_dim:]  # (bs, 1)
                predicted_reward_logvar = logvar[:, :, self.obs_dim:]  # (bs, 1)
        
                # Uncertainty-weighted reconstruction loss
                obs_inv_var = torch.exp(-decoded_obs_logvar)
                recon_loss = (torch.pow(decoded_obs_mean - s, 2)*obs_inv_var).mean() + decoded_obs_logvar.mean()
                
                # KL divergence loss
                kl_loss = self._kl_divergence(posterior_mean, posterior_logvar, prior_mean, prior_logvar)
                
                # Reward prediction loss
                rew_inv_var = torch.exp(-predicted_reward_logvar)
                reward_loss = (torch.pow(predicted_reward_mean - r, 2)*rew_inv_var).mean() + predicted_reward_logvar.mean()
                
                # Total loss
                loss = recon_loss + 0.1 * kl_loss + reward_loss

                # backward
                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.set_postfix(
                    train_loss=loss.item(),
                    recon_loss=recon_loss.item(),
                    kl_loss=kl_loss.item(),
                    reward_loss=reward_loss.item(),
                    holdout_loss=np.mean(holdout_losses)
                )

            new_val_losses, improve_ks = [], []
            for k in range(1, max_adm_step+1):
                k_step_seq = buffer.sample_all_nstep(k, start_idx=train_size)
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
    
    def _kl_divergence(self, mu1, logvar1, mu2, logvar2):
        """ Compute KL divergence between two Gaussian distributions """
        kl_loss = 0.5 * torch.sum(
            logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2).pow(2)) / torch.exp(logvar2) - 1
        )
        return kl_loss
    
    def validate_from(self, s, a, r, s_):
        """ validate any-step dynamics model (fixed k-step validation) """
        trgt = torch.cat((s_, r), dim=-1)
        mean, _ = self.forward(s, a)
        loss = ((mean - trgt) ** 2).mean()
        return float(loss.cpu().detach().numpy())
    
    def save_model(self, filepath):
        """ save model """
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """ load model """
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)

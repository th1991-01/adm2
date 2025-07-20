import os
import copy
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

from .ensemble_model import EnsembleDynamicsModel

class EnsembleDynamics(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        device: str = "cuda:0"
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = self.obs_dim + 1
        self.device = device
        
        self.model = EnsembleDynamicsModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            output_dim=self.output_dim,
            device=device
        )
        
        # 'mean' and 'std' for normalization
        self.register_parameter("obs_mu", nn.Parameter(torch.zeros(self.obs_dim), requires_grad=False))
        self.register_parameter("obs_std", nn.Parameter(torch.zeros(self.obs_dim), requires_grad=False))
        self.register_parameter("act_mu", nn.Parameter(torch.zeros(self.action_dim), requires_grad=False))
        self.register_parameter("act_std", nn.Parameter(torch.zeros(self.action_dim), requires_grad=False))
        
        self.to(self.device)
        
    def set_mu_std(self, obs_mu, obs_std, act_mu, act_std):
        self.obs_mu.data = torch.as_tensor(obs_mu, dtype=torch.float32, device=self.device)
        self.obs_std.data = torch.as_tensor(obs_std, dtype=torch.float32, device=self.device)
        self.act_mu.data = torch.as_tensor(act_mu, dtype=torch.float32, device=self.device)
        self.act_std.data = torch.as_tensor(act_std, dtype=torch.float32, device=self.device)
        
    def forward(self, obs, action):
        # shape@obs: (bs, obs_dim)
        # shape@actions: (bs, action_dim)
        _obs = (obs - self.obs_mu) / self.obs_std
        _action = (action - self.act_mu) / self.act_std
        obs_act = torch.cat((_obs, _action), dim=-1)
        mean, logvar = self.model(obs_act)
        return mean, logvar
    
    @ torch.no_grad()
    def dyna_dist(self, obs, action):
        # obs: (bs, -1)
        # action: (bs, -1)
        mean, logvar = self.forward(obs, action)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))
        
        mean = mean[self.model.elites]
        std = std[self.model.elites]
        
        next_obs_mean = mean[..., :-1]
        next_obs_std = std[..., :-1]
        reward_mean = mean[..., -1:]
        reward_std = std[..., -1:]
        return next_obs_mean, next_obs_std, reward_mean, reward_std

    def format_samples_for_training(self, data: Dict):
        s = data["s"]
        a = data["a"]
        s_ = data["s_"]
        r = data["r"]
        delta_s = s_ - s
        inputs = torch.cat((s, a), dim=-1)
        targets = torch.cat((delta_s, r), dim=-1)
        return inputs, targets
    
    def learn_from(self, buffer, lr, batch_size, max_holdout=1000, min_epochs=1):
        """ learn ensemble dynamics model """
        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        
        # set mean and std
        obs_mu, obs_std, act_mu, act_std = buffer.cal_mu_std()
        self.set_mu_std(obs_mu, obs_std, act_mu, act_std)

        data_size = buffer.size
        holdout_size = min(int(data_size * 0.2), max_holdout)
        train_size = data_size - holdout_size
        
        all_data = buffer.sample_all()
        inputs, targets = self.format_samples_for_training(all_data)
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]
        
        data_idxes = torch.randint(train_size, (self.model.num_ensemble, train_size))
        def shuffle_rows(arr):
            random_vals = torch.rand_like(arr.float())
            idxes = torch.argsort(random_vals, dim=-1)
            batch_indices = torch.arange(arr.size(0)).unsqueeze(1).expand(-1, arr.size(1))
            return arr[batch_indices, idxes]

        epoch = 0
        holdout_losses = [1e10] * self.model.num_ensemble
        cnt = 0

        while True:
            epoch += 1
            epoch_inputs = train_inputs[data_idxes]
            epoch_targets = train_targets[data_idxes]
            data_idxes = shuffle_rows(data_idxes)

            pbar = tqdm(range(train_size//batch_size), desc=f"[M][Epoch {epoch} @ Ensemble Dynamics Model Training]")
            for batch_id in pbar:
                inputs_batch = epoch_inputs[:, batch_id * batch_size:(batch_id + 1) * batch_size]
                targets_batch = epoch_targets[:, batch_id * batch_size:(batch_id + 1) * batch_size]
            
                mean, logvar = self.forward(inputs_batch[..., :self.obs_dim], inputs_batch[..., self.obs_dim:])
                inv_var = torch.exp(-logvar)
                # Average over batch and dim, sum over ensembles.
                mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
                var_loss = logvar.mean(dim=(1, 2))
                loss = mse_loss_inv.sum() + var_loss.sum()
                loss = loss + self.model.get_decay_loss()
                loss = loss + 0.01 * self.model.max_logvar.sum() - 0.01 * self.model.min_logvar.sum()

                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.set_postfix(
                    train_loss=loss.item(),
                    holdout_loss=np.mean(holdout_losses),
                )
                
            new_holdout_losses = self.validate_from(holdout_inputs, holdout_targets)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= 5) and epoch >= min_epochs:
                break
            
        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        return [float(loss) for loss in holdout_losses]
    
    @ torch.no_grad()
    def validate_from(self, inputs, targets):
        self.model.eval()
        mean, _ = self.forward(inputs[..., :self.obs_dim], inputs[..., self.obs_dim:])
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save_model(self, filepath):
        """ save model """
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """ load model """
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)

import torch
import torch.nn as nn
import numpy as np

from components import ACTOR, CRITIC

class PPOAgent:
    """ proximal policy optimization """

    def __init__(
        self, 
        obs_shape, 
        hidden_dims, 
        action_dim,
        action_space,
        actor_lr,
        critic_lr,
        clip_ratio=0.2,
        value_clip=False,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        ppo_epoch=5,
        mini_batch_size=64,
        device="cuda:0"
    ):
        # actor
        self.actor = ACTOR["prob"](obs_shape, hidden_dims, action_dim).to(device)

        # critic
        self.critic = CRITIC["v"](obs_shape, hidden_dims).to(device)

        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # action space
        self.action_space = action_space

        # other parameters
        self._clip_ratio = clip_ratio
        self._value_clip = value_clip
        self._value_coef = value_coef
        self._entropy_coef = entropy_coef
        self._ppo_epoch = ppo_epoch
        self._mini_batch_size = mini_batch_size
        self._max_grad_norm = max_grad_norm
        self._eps = np.finfo(np.float32).eps.item()
        self.device = device

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def actor4ward(self, obs, deterministic=False):
        """ forward propagation of actor """
        dist = self.actor(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs)

        action_scale = torch.tensor((self.action_space.high-self.action_space.low)/2, device=self.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale*(1-squashed_action.pow(2))+self._eps).sum(-1, keepdim=True)

        return action_scale*squashed_action, log_prob, value
    
    def logprob_of(self, dist, squashed_action):
        action_scale = torch.tensor((self.action_space.high-self.action_space.low)/2, device=self.device)
        squashed_action /= action_scale
        src_action = torch.arctanh(squashed_action)
        log_prob = dist.log_prob(src_action)
        log_prob = log_prob - torch.log(action_scale*(1-squashed_action.pow(2))+self._eps).sum(-1, keepdim=True)
        return log_prob
    
    def act_and_value(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob, value = self.actor4ward(obs, deterministic)
        return action, log_prob, value

    def act(self, obs, deterministic=False):
        """ sample action """
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, _, _ = self.actor4ward(obs, deterministic)
        return action
    
    def learn(self, s, a, log_prob, v, adv, ret):
        policy_losses, value_losses, entropy_losses, kl, values_ = [], [], [], [], []
        
        # normalize adv
        adv_mean, adv_std = adv.mean(), adv.std()
        adv = (adv - adv_mean) / (adv_std + 1e-5)

        batch_size = s.shape[0]
        for i in range(self._ppo_epoch):
            indexes = np.random.permutation(np.arange(batch_size))
            for mini_batch_num in range(int(np.ceil(len(indexes) / self._mini_batch_size))):
                index = indexes[mini_batch_num * self._mini_batch_size:(mini_batch_num + 1) * self._mini_batch_size]
                obs, actions, returns, value_preds, advs, logp_olds = \
                    s[index], a[index], ret[index], v[index], adv[index], log_prob[index]
                
                # policy loss
                dist = self.actor(obs)
                log_probs = self.logprob_of(dist, actions)
                ratio = torch.exp(log_probs - logp_olds)
                clip_adv = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advs
                policy_loss = -(torch.min(ratio * advs, clip_adv)).mean()
                approx_kl = (logp_olds - log_probs).mean()

                # value loss
                values = self.critic(obs)
                if self._value_clip:
                    value_pred_clipped = value_preds + \
                        (values - value_preds).clamp(-self._clip_ratio, self._clip_ratio)
                    value_loss = (values - returns).pow(2)
                    value_loss_clipped = (value_pred_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (values - returns).pow(2).mean()

                # entropy loss
                entropy_loss = dist.entropy().mean()

                total_loss = policy_loss + self._value_coef * value_loss - self._entropy_coef * entropy_loss

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())+list(self.critic.parameters()),
                    self._max_grad_norm
                )
                self.actor_optim.step()
                self.critic_optim.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl.append(approx_kl.item())
                values_.append(values.cpu().detach().numpy().mean())

        info = {
            "loss": {
                "actor": float(np.mean(policy_losses)),
                "critic": float(np.mean(value_losses))
            },
            "kl": float(np.mean(kl)),
            "value": float(np.mean(values_))
        }

        if self._entropy_coef:
            info["loss"]["entropy"] = float(np.mean(entropy_losses))
        
        return info

    def save_model(self, filepath):
        """ save model """
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        torch.save(state_dict, filepath)

    def load_model(self, filepath):
        """ load model """
        state_dict = torch.load(filepath)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])

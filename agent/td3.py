import torch
import numpy as np
from copy import deepcopy

from components import ACTOR, CRITIC

class TD3Agent:
    """ td3 """

    def __init__(
        self, 
        obs_shape, 
        hidden_dims, 
        action_dim,
        action_space,
        actor_lr,
        critic_lr,
        tau=0.005, 
        gamma=0.99,
        q_clip=None,
        explore_noise=0.25,
        target_policy_noise=0.2,
        noise_clip=0.5,
        device="cuda:0"
    ):
        # actor
        self.max_action = action_space.high[0]
        self.actor = ACTOR["deter"](obs_shape, hidden_dims, action_dim, self.max_action).to(device)

        # critic
        self.critic1 = CRITIC["q"](obs_shape, hidden_dims, action_dim).to(device)
        self.critic2 = CRITIC["q"](obs_shape, hidden_dims, action_dim).to(device)
        # target critic
        self.critic1_trgt = deepcopy(self.critic1)
        self.critic2_trgt = deepcopy(self.critic2)
        self.critic1_trgt.eval()
        self.critic2_trgt.eval()

        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # actor update frequency
        self.actor_freq = 2
        self.critic_cnt = 0

        # other parameters
        self._tau = tau
        self._gamma = gamma
        self._q_clip = q_clip
        self._eps = np.finfo(np.float32).eps.item()
        self.explore_noise = explore_noise * self.max_action
        self.trgt_pi_noise = target_policy_noise * self.max_action
        self.noise_clip = noise_clip * self.max_action
        self.device = device

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self):
        """ synchronize weight """
        for trgt, src in zip(self.critic1_trgt.parameters(), self.critic1.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self._tau) + src.data*self._tau)
        for trgt, src in zip(self.critic2_trgt.parameters(), self.critic2.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self._tau) + src.data*self._tau)

    def actor4ward(self, obs, deterministic=False):
        """ forward propagation of actor """
        action = self.actor(obs)
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)
            action = action.clamp(-self.max_action, self.max_action)
        return action

    def act(self, obs, deterministic=False):
        """ sample action """
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action = self.actor4ward(obs, deterministic)
        return action

    def learn(self, s, a, r, s_, done):
        """ learn from (s, a, r, s_, done) """
        s    = torch.as_tensor(s, device=self.device)
        a    = torch.as_tensor(a, device=self.device)
        r    = torch.as_tensor(r, device=self.device)
        s_   = torch.as_tensor(s_, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        # update critic
        q1, q2 = self.critic1(s, a).flatten(), self.critic2(s, a).flatten()
        with torch.no_grad():
            a_ = self.actor4ward(s_, deterministic=True)
            a_ += (self.trgt_pi_noise*torch.randn_like(a_)).clamp(-self.noise_clip, self.noise_clip)
            a_ = a_.clamp(-self.max_action, self.max_action)
            q_ = torch.min(self.critic1_trgt(s_, a_), self.critic2_trgt(s_, a_))
            if self._q_clip is not None:
                q_ = q_.clip(None, self._q_clip)
            q_trgt = r.flatten() + self._gamma*(1-done.flatten())*q_.flatten()

        critic1_loss = ((q1-q_trgt).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2-q_trgt).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self.critic_cnt += 1

        actor_loss = None
        if self.critic_cnt % self.actor_freq == 0:
            # update actor
            a = self.actor4ward(s, deterministic=True)
            q1, q2 = self.critic1(s, a).flatten(), self.critic2(s, a).flatten()
            actor_loss = (-torch.min(q1, q2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        # synchronize weight
        self._sync_weight()

        info = {
            "loss": {
                "actor": actor_loss.item() if actor_loss else None,
                "critic1": critic1_loss.item(),
                "critic2": critic2_loss.item()
            }
        }

        return info

    def save_model(self, filepath):
        """ save model """
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
        }
        torch.save(state_dict, filepath)

    def load_model(self, filepath):
        """ load model """
        state_dict = torch.load(filepath)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])

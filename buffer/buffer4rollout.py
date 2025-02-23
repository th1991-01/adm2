import torch
import numpy as np
import scipy.signal

class RolloutBuffer:
    def __init__(
        self,
        n_envs,
        n_steps,
        obs_shape,
        action_dim,
        gamma=0.99,
        gae_lambda=0.95,
        device="cuda"
    ):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self.device = torch.device(device)
        self.reset()
    
    def reset(self):
        self.observations = torch.zeros((self.n_envs, self.n_steps,) + self.obs_shape, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.n_envs, self.n_steps, self.action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.n_envs, self.n_steps, 1), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.n_envs, self.n_steps, 1), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((self.n_envs, self.n_steps, 1), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.n_envs, self.n_steps, 1), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((self.n_envs, self.n_steps, 1), dtype=torch.float32, device=self.device)
        self._ptr = 0
        self._episode_start_ids = np.zeros(self.n_envs, dtype=np.int64)
    
    def store_batch(self, obs, action, reward, value, log_prob):
        assert self._ptr < self.n_steps
        self.observations[:, self._ptr] = obs
        self.actions[:, self._ptr] = action
        self.rewards[:, self._ptr] = reward
        self.values[:, self._ptr] = value
        self.log_probs[:, self._ptr] = log_prob
        self._ptr += 1
    
    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_episode(self, env_ids, last_values):
        last_values = last_values.detach().cpu().numpy()
        
        for it in range(len(env_ids)):
            env_id = env_ids[it]
            last_value = last_values[it]
            
            episode_slice = slice(self._episode_start_ids[env_id], self._ptr)
            rews = np.append(self.rewards[env_id, episode_slice].cpu().numpy(), last_value)
            vals = np.append(self.values[env_id, episode_slice].cpu().numpy(), last_value)

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self._gamma * vals[1:] - vals[:-1]
            self.advantages[env_id, episode_slice] = torch.as_tensor(
                self._discount_cumsum(deltas, self._gamma * self._gae_lambda).reshape(-1, 1).copy(),
                device=self.device
            )

            # the next line computes rewards-to-go, to be targets for the value function
            self.returns[env_id, episode_slice] = torch.as_tensor(
                self._discount_cumsum(rews, self._gamma)[:-1].reshape(-1, 1).copy(),
                device=self.device
            )

            self._episode_start_ids[env_id] = self._ptr
    
    def sample_all(self):
        assert self._ptr == self.n_steps
        self._ptr = 0
        self._episode_start_ids = np.zeros(self.n_envs, dtype=np.int64)

        return {
            "s": self.observations.reshape(self.n_envs*self.n_steps, -1),
            "a": self.actions.reshape(self.n_envs*self.n_steps, -1),
            "ret": self.returns.reshape(self.n_envs*self.n_steps, -1),
            "v": self.values.reshape(self.n_envs*self.n_steps, -1),
            "adv": self.advantages.reshape(self.n_envs*self.n_steps, -1),
            "log_prob": self.log_probs.reshape(self.n_envs*self.n_steps, -1)
        }

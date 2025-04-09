import torch
import numpy as np
from tqdm import tqdm

class ReplayBuffer:
    """ general replay buffer """

    def __init__(self, buffer_size, obs_shape, action_dim, device="cuda"):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.capacity = buffer_size
        self.device = device
        self.reset()

    def reset(self):
        self.size = 0
        self.cnt = 0

        self.memory = {
            "s":       torch.zeros((self.capacity, *self.obs_shape), dtype=torch.float32, device=self.device),
            "a":       torch.zeros((self.capacity, self.action_dim), dtype=torch.float32, device=self.device),
            "r":       torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device),
            "s_":      torch.zeros((self.capacity, *self.obs_shape), dtype=torch.float32, device=self.device),
            "done":    torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device),
            "timeout": torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        }

        # not used
        self.cur_epi_start = 0

    def store(self, s, a, r, s_, done, timeout):
        """ store transition (s, a, r, s_, done, timeout) """
        self.memory["s"][self.cnt] = s
        self.memory["a"][self.cnt] = a
        self.memory["r"][self.cnt] = r
        self.memory["s_"][self.cnt] = s_
        self.memory["done"][self.cnt] = done
        self.memory["timeout"][self.cnt] = timeout

        self.cnt = (self.cnt+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def store_batch(self, s, a, r, s_, done, timeout):
        """ store batch transitions (s, a, r, s_, done, timeout) """
        batch_size = len(s)

        indices = torch.arange(self.cnt, self.cnt+batch_size)%self.capacity
        self.memory["s"][indices] = s
        self.memory["a"][indices] = a
        self.memory["r"][indices] = r
        self.memory["s_"][indices] = s_
        self.memory["done"][indices] = done
        self.memory["timeout"][indices] = timeout

        self.cnt = (self.cnt+batch_size)%self.capacity
        self.size = min(self.size+batch_size, self.capacity)

    def load_dataset(self, dataset, rew_bias=0.0):
        """ load dataset """
        N = dataset["rewards"].shape[0]
        if self.capacity < N:
            self.capacity = N
            self.reset()
            
        self.memory["s"][:N] = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=self.device)
        self.memory["a"][:N] = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=self.device)
        self.memory["r"][:N] = torch.as_tensor(dataset["rewards"], dtype=torch.float32, device=self.device).reshape(-1, 1) + rew_bias
        self.memory["s_"][:N] = torch.as_tensor(dataset["next_observations"], dtype=torch.float32, device=self.device)
        self.memory["done"][:N] = torch.as_tensor(dataset["terminals"], dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.memory["timeout"][:N] = torch.as_tensor(dataset["timeouts"], dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.cnt = N
        self.size = N

    def load_neorl_dataset(self, dataset, rew_bias=0.0):
        """ load neorl dataset """
        N = dataset["reward"].shape[0]
        if self.capacity < N:
            self.capacity = N
            self.reset()

        start_indexes = dataset["index"]
        for i in tqdm(range(N-1), desc="Loading dataset (in tensor)"):
            obs = torch.as_tensor(dataset["obs"][i], dtype=torch.float32, device=self.device)
            next_obs = torch.as_tensor(dataset["next_obs"][i], dtype=torch.float32, device=self.device)
            action = torch.as_tensor(dataset["action"][i], dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(dataset["reward"][i], dtype=torch.float32, device=self.device)
            done = bool(dataset["done"][i])
            timeout = (i + 1 in start_indexes)
            self.store(obs, action, reward + rew_bias, next_obs, done, timeout)

    def cal_mu_std(self):
        """ calculate mean and std of obs and action """
        obs_mu = torch.mean(self.memory["s"][:self.size], dim=0)
        obs_std = torch.std(self.memory["s"][:self.size], dim=0)
        obs_std[obs_std < 1e-12] = 1.0
        act_mu = torch.mean(self.memory["a"][:self.size], dim=0)
        act_std = torch.std(self.memory["a"][:self.size], dim=0)
        act_std[act_std < 1e-12] = 1.0
        return obs_mu, obs_std, act_mu, act_std

    def sample(self, batch_size):
        """ sample a batch of data """
        indices = np.random.randint(0, self.size, batch_size)
        return {var: self.memory[var][indices] for var in self.memory.keys()}

    def sample_all(self):
        """ sample all data """
        indices = np.arange(self.size)
        return {var: self.memory[var][indices] for var in self.memory.keys()}

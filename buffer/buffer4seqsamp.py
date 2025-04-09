import numpy as np
from tqdm import tqdm
from .buffer import ReplayBuffer

class ReplayBufferForSeqSampling(ReplayBuffer):
    """ replay buffer for sequential action sampling """

    def __init__(self, buffer_size, obs_shape, action_dim, device="cuda"):
        super().__init__(buffer_size, obs_shape, action_dim, device)
        self.reset()

    def reset(self):
        super().reset()
        self.dist_from_end = np.zeros(self.capacity, dtype=np.float32)
        self.cur_epi_start = 0

    def store(self, s, a, r, s_, done, timeout):
        """ store transition (s, a, r, s_, done, timeout) """
        super().store(s, a, r, s_, done, timeout)
        if self.cur_epi_start < self.cnt:
            self.dist_from_end[self.cur_epi_start:self.cnt] += 1
        else:
            # refresh from the head of the queue
            self.dist_from_end[self.cur_epi_start:] += 1
            self.dist_from_end[:self.cnt] += 1
        if done == 1 or timeout == 1: self.cur_epi_start = self.cnt
        
    def load_dataset(self, dataset, reward_bias=0.0):
        """ load dataset """
        super().load_dataset(dataset, reward_bias)
        self.dist_from_end = np.zeros(self.capacity, dtype=np.float32)
        self.cur_epi_start = 0
        for i in tqdm(range(self.cnt), desc="Preparing dataset"):
            self.dist_from_end[self.cur_epi_start:i+1] += 1
            if self.memory["done"][i].item() == 1: self.cur_epi_start = i + 1

    def sample_nstep(self, batch_size, nstep, start_idx=None, end_idx=None):
        """ sample a batch of {nstep} data """
        if start_idx == None: start_idx = 0
        if end_idx == None: end_idx = self.size

        all_start_indices = np.arange(start_idx, end_idx)[self.dist_from_end[start_idx:end_idx]>=nstep]
        start_indices = np.random.choice(all_start_indices, batch_size)
        indices = (start_indices.reshape(-1, 1) + np.arange(nstep))%self.size
        return {var: self.memory[var][indices] for var in self.memory.keys()}
    
    def sample_all_nstep(self, nstep, start_idx=None, end_idx=None):
        """ sample all {nstep} data """
        if start_idx == None: start_idx = 0
        if end_idx == None: end_idx = self.size

        start_indices = np.arange(start_idx, end_idx)[self.dist_from_end[start_idx:end_idx]>=nstep]
        indices = (start_indices.reshape(-1, 1) + np.arange(nstep))%self.size
        return {var: self.memory[var][indices] for var in self.memory.keys()}

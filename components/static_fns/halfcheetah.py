import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
        done = ~not_done
        done = done[:, None]
        return done
    
    @staticmethod
    def reward_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        
        x_vel = next_obs[:, 8]
        cost = np.sum(np.square(act), axis=-1)
        reward = 1.0 * x_vel - 0.1 * cost
        reward = reward[:, None]
        return reward

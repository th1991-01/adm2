import numpy as np

class StaticFns:
    
    @staticmethod
    def healthy_fn(obs):
        assert len(obs.shape) == 2

        height = obs[:, 0]
        angle = obs[:, 1]
        healthy = np.isfinite(obs).all(axis=-1) \
                    * np.abs(obs[:, 1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)
        return healthy
    
    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        not_done = StaticFns.healthy_fn(next_obs)
        done = ~not_done
        done = done[:, None]
        return done
    
    @staticmethod
    def reward_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        x_vel = next_obs[:, 5]
        healthy = StaticFns.healthy_fn(next_obs)
        cost = np.sum(np.square(act), axis=-1)
        reward = 1.0 * healthy + 1.0 * x_vel - 1e-3 * cost
        reward = reward[:, None]
        return reward

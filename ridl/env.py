import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class ConvDiffusionEnv(gym.Env):
    def __init__(self, df, state_feats, action_feats, raw_images, transform=None):
        super().__init__()
        self.df           = df.reset_index(drop=True)
        self.state_feats  = state_feats
        self.act_feats    = action_feats
        self.raw_images   = raw_images       # list/array of H×W×C uint8 frames
        self.transform    = transform        # TorchVision or custom preprocessing
        self.max_steps    = len(df) - 1
        self.current_step = 0

        # numeric part
        lo_num = df[state_feats].min().astype(np.float32).values
        hi_num = df[state_feats].max().astype(np.float32).values
        num_space = spaces.Box(lo_num, hi_num, dtype=np.float32)

        # image part
        h, w, c = raw_images[0].shape
        img_space = spaces.Box(0, 255, shape=(h, w, c), dtype=np.uint8)

        # combine into a Dict space
        self.observation_space = spaces.Dict({
            'numeric': num_space,
            'image':   img_space
        })

        lo_act = df[action_feats].min().astype(np.float32).values
        hi_act = df[action_feats].max().astype(np.float32).values
        self.action_space = spaces.Box(lo_act, hi_act, dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        num_obs = self.df[self.state_feats].iloc[0].astype(np.float32).values
        img_obs = self.raw_images[0]
        return {'numeric': num_obs, 'image': img_obs}, {}

    def step(self, action):
        # clip against .low and .high, not .bounds
        action = np.clip(action,
                         self.action_space.low,
                         self.action_space.high)

        rew = float(self.df['Reward'].iloc[self.current_step])
        self.current_step += 1
        done = self.current_step >= self.max_steps
        idx  = min(self.current_step, self.max_steps)

        obs = self.df[self.state_feats].iloc[idx].astype(np.float32).values
        return obs, rew, done, False, {}

    def render(self, mode='human'):
        """If human: show via plt; if 'rgb_array': return the raw frame."""
        frame = self.raw_images[self.current_step]
        if mode == 'rgb_array':
            return frame
        elif mode == 'human':
            plt.imshow(frame)
            plt.title(f"Step {self.current_step}")
            plt.axis('off')
            plt.show()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

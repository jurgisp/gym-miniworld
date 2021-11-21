import time
from datetime import datetime
from pathlib import Path

import gym
import numpy as np


class ActionRewardResetWrapper(gym.Wrapper):
    """From pydreamer.envs.wrappers."""

    def __init__(self, env, no_terminal=False):
        super().__init__(env)
        self.env = env
        self.no_terminal = no_terminal
        # Handle environments with one-hot or discrete action, but collect always as one-hot
        self.action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(action, int):
            action_vec = np.zeros(self.action_size)
            action_vec[action] = 1.0
        else:
            assert isinstance(action, np.ndarray) and action.shape == (self.action_size,), "Wrong one-hot action shape"
            action_vec = action
        obs['action'] = action_vec
        obs['reward'] = np.array(reward)
        obs['terminal'] = np.array(False if self.no_terminal or info.get('time_limit') else done)
        obs['reset'] = np.array(False)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs['action'] = np.zeros(self.action_size)
        obs['reward'] = np.array(0.0)
        obs['terminal'] = np.array(False)
        obs['reset'] = np.array(True)
        return obs


class CollectWrapper(gym.Wrapper):
    """From pydreamer.envs.wrappers."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.episode = []
        self.episode_id = ''

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode.append(obs.copy())
        if done:
            episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
            info['episode'] = episode
        info['episode_id'] = self.episode_id
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.episode = [obs.copy()]
        self.episode_id = datetime.now().strftime('%Y%m%dT%H%M%S')
        return obs


class SaveNpzWrapper(gym.Wrapper):

    def __init__(self, env, log_dir):
        super().__init__(env)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        data = info.get('episode')
        if data:
            self.save_npz(data, info['episode_id'])
        return obs, reward, done, info

    def save_npz(self, data, episode_id):
        path = self.log_dir / (self.episode_name(data, episode_id) + '.npz')
        with path.open('wb') as f:
            np.savez_compressed(f, **data)
        print(f'Saved {path}', {k: v.shape for k, v in data.items()})

    def episode_name(self, data, episode_id):
        reward = data['reward'].sum()
        steps = len(data['reward']) - 1
        return f'{episode_id}-r{reward:.0f}-{steps:04}'


class LogCsvWrapper(gym.Wrapper):

    def __init__(self, env, log_path, rewards_only=True):
        super().__init__(env)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rewards_only = rewards_only
        self.episode_start = 0
        self.last_reward_step = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.append_csv(obs, reward, done, info)
        return obs, reward, done, info

    def reset(self):
        self.episode_start = time.time()
        self.last_reward_step = 0
        return self.env.reset()

    def append_csv(self, obs, reward, done, info):
        if self.rewards_only and reward < 1.0:
            return  # Only log reward steps, for manual inspection 
        
        step = self.step_count
        episode_id = info.get('episode_id')
        elapsed_time = time.time() - self.episode_start
        steps_since_reward = step - self.last_reward_step
        self.last_reward_step = step

        with self.log_path.open('a') as f:
            f.write(f'{episode_id}\t{step}\t{reward}\t{steps_since_reward}\t{elapsed_time:.0f}\t{self.spec.id}\t{self.max_episode_steps}\n')


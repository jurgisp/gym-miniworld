import math
import numpy as np
import gym


class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


class GreyscaleWrapper(gym.ObservationWrapper):
    """
    Convert image obserations from RGB to greyscale
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[0], obs_shape[1], 1],
            dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        obs = (
            0.30 * obs[:, :, 0] +
            0.59 * obs[:, :, 1] +
            0.11 * obs[:, :, 2]
        )

        return np.expand_dims(obs, axis=2)


class MapWrapper(gym.ObservationWrapper):
    """
    Include top-down map as observation
    """

    def __init__(self, env=None):
        super().__init__(env)
        # TODO
        # self.observation_space = ...

    def observation(self, obs_img):
        obs = {}
        obs['image'] = obs_img
        obs['map_agent'] = self.get_map(with_agent=True)
        obs['map_centered'] = self.get_map(centered=True)
        return obs

    def get_map(self, with_agent=False, centered=False):
        env = self.env
        assert env.gap_size == env.room_size
        s = env.room_size
        assert env.max_x - env.min_x == env.max_z - env.min_z
        n = round((env.max_x - env.min_x) / s)

        # Use the same categorical values as MiniGrid
        #   GRID_VALUES = np.array([
        #       [0, 0, 0],  # Invisible
        #       [1, 0, 0],  # Empty
        #       [2, 5, 0],  # Wall
        #       [8, 1, 0],  # Goal
        #       [10, 0, 0], # agent(dir=0)
        #       [10, 0, 1], # agent(dir=1)
        #       [10, 0, 2], # agent(dir=2)
        #       [10, 0, 3], # agent(dir=3)
        #       ...

        map = np.zeros((n, n), dtype=int)
        map[:, :] = 2  # wall
        for room in env.rooms:
            ix = round((room.min_x - env.min_x) / s)
            iz = round((room.min_z - env.min_z) / s)
            map[ix, iz] = 1  # empty

        agent_ix = int(np.floor((env.agent.pos[0] - env.min_x) / 2))
        agent_iz = int(np.floor((env.agent.pos[2] - env.min_x) / 2))
        agent_dir = round(env.agent.dir / (np.pi / 2)) % 4
        if with_agent:
            map[agent_ix, agent_iz] = 4 + agent_dir  # agent(dir)

        if centered:
            # Bigger [2n-1;2n-1] map, where agent is positioned in the center at [n-1;n-1]
            nc = 2 * n - 1
            mapc = np.zeros((nc, nc), dtype=int)
            mapc[:, :] = 2  # wall
            mapc[n - 1 - agent_ix: 2 * n - 1 - agent_ix, n - 1 - agent_iz: 2 * n - 1 - agent_iz] = map
            mapc = np.rot90(mapc, agent_dir)
            map = mapc

        return map

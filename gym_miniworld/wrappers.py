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


class DictWrapper(gym.ObservationWrapper):
    """
    Return dictionary observations {'image': obs}
    """

    def __init__(self, env=None):
        super().__init__(env)
        # self.observation_space = ...  # TODO

    def observation(self, obs_img):
        return {'image': obs_img}


class MapWrapper(gym.Wrapper):
    """
    Include top-down map as observation
    """

    def __init__(self, env=None):
        super().__init__(env)
        self._reset_map_seen()
        # self.observation_space = ...

    @property
    def map_size(self):
        env = self.env
        s = env.room_size
        assert env.gap_size == env.room_size
        assert env.max_x - env.min_x == env.max_z - env.min_z
        n = round((env.max_x - env.min_x) / s)
        return n

    @property
    def agent_pos(self):
        env = self.env
        s = env.room_size
        agent_ix = int(np.floor((env.agent.pos[0] - env.min_x) / s))
        agent_iz = int(np.floor((env.agent.pos[2] - env.min_x) / s))
        agent_dir = round(env.agent.dir / (np.pi / 2)) % 4  # counter-clockwise
        return agent_ix, agent_iz, agent_dir

    def _reset_map_seen(self):
        n = self.map_size
        self._map_seen = np.zeros((n, n), dtype=bool)

    def _update_map_seen(self):
        n = self.map_size
        ix, iz, _ = self.agent_pos
        self._map_seen[ix, iz] = True
        # "See" 4 directly adjancent cells
        for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ix1, iz1 = ix + dx, iz + dz
            if ix1 >= 0 and iz1 >= 0 and ix1 < n and iz1 < n:
                self._map_seen[ix1, iz1] = True

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._reset_map_seen()
        self._update_map_seen()
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._update_map_seen()
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        obs['map'] = self.get_map(only_seen=True)
        obs['map_agent'] = self.get_map(with_agent=True)
        # obs['map_centered'] = self.get_map(centered=True)
        # print(obs['map'].T)
        return obs

    def get_map(self, with_agent=False, centered=False, only_seen=False):
        env = self.env
        s = self.room_size
        n = self.map_size

        # Use the same categorical values as MiniGrid
        #   GRID_VALUES = np.array([
        #       [0, 0, 0],  # Invisible
        #       [1, 0, 0],  # Empty
        #       [2, 5, 0],  # Wall
        #       [8, 1, 0],  # Goal
        #       [10, 0, 0], # agent(dir=right)
        #       [10, 0, 1], # agent(dir=down)
        #       [10, 0, 2], # agent(dir=left)
        #       [10, 0, 3], # agent(dir=up)
        #       ...

        map = np.zeros((n, n), dtype=int)
        map[:, :] = 2  # wall

        for room in env.rooms:
            ix = round((room.min_x - env.min_x) / s)
            iz = round((room.min_z - env.min_z) / s)
            map[ix, iz] = 1  # empty

        for ent in env.entities:
            ix = int(np.floor((ent.pos[0] - env.min_x) / s))
            iz = int(np.floor((ent.pos[2] - env.min_x) / s))
            if ent == env.agent:
                if with_agent:
                    dir = round(ent.dir / (np.pi / 2)) % 4  # counter-clockwise
                    minigrid_agent_dir = (8 - dir) % 4  # MiniGrid's agent_dir goes in clockwise direction
                    map[ix, iz] = 4 + minigrid_agent_dir
            else:
                map[ix, iz] = 3  # assume goal

        if only_seen:
            map = map * self._map_seen

        if centered:
            agent_ix, agent_iz, agent_dir = self.agent_pos
            # Bigger [2n-1;2n-1] map, where agent is positioned in the center at [n-1;n-1]
            nc = 2 * n - 1
            mapc = np.zeros((nc, nc), dtype=int)
            mapc[:, :] = 2  # wall
            mapc[n - 1 - agent_ix: 2 * n - 1 - agent_ix, n - 1 - agent_iz: 2 * n - 1 - agent_iz] = map
            mapc = np.rot90(mapc, agent_dir - 1)  # forward is up
            map = mapc

        return map


class AgentPosWrapper(gym.ObservationWrapper):
    """
    Include agents global position
    """

    def __init__(self, env=None):
        super().__init__(env)
        # self.observation_space = ...  # TODO

    def observation(self, obs):
        pos = self.env.agent.pos
        dir = self.env.agent.dir
        room_size = self.env.room_size
        obs['agent_pos'] = np.round(np.array([pos[0], pos[2]]) / room_size, 5)             # (x,y,z) => (x,z)
        obs['agent_dir'] = np.round(np.array([np.cos(dir), -np.sin(dir)]), 5)  # angle => (dx,dz)
        return obs


class PixelMapWrapper(gym.ObservationWrapper):
    """
    Include agent-centric pixel map as observation
    """

    def __init__(self, env=None):
        from skimage.transform import resize
        super().__init__(env)
        self._resize_fn = resize
        # self.observation_space = ...  # TODO

    def observation(self, obs):
        # obs['map'] = self.get_map(with_agent=Fals)
        # obs['map_agent'] = self.get_map()
        obs['map_centered'] = self.get_map(centered=True)
        return obs

    def get_map(self, centered=False):
        env = self.env
        if centered:
            map = env.render_top_view_centered()
        else:
            map = env.render_top_view()
        map = self._resize_fn(map, (64, 64), anti_aliasing=True)
        return map

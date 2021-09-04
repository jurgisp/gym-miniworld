import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame
from ..params import DEFAULT_PARAMS


class MazeBase(MiniWorldEnv):
    """
    Base env for making mazes easier
    """

    def __init__(
        self,
        size=9,
        room_size=3,
        max_steps=1500,
        forward_step_rooms=1.0,
        turn_step=90,
    ):
        self.size = size
        self.room_size = self.gap_size = room_size
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step_rooms * room_size)
        params.set('turn_step', turn_step)
        super().__init__(max_steps, params=params, obs_width=64, obs_height=64)
        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        raise NotImplementedError

    def _gen_map_world(self, map: np.ndarray):
        assert map.shape == (self.size, self.size), 'Wrong map size'

        # map = np.array([
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        # ])

        rooms = [
            [None for _ in range(self.size)]
            for _ in range(self.size)
        ]

        # Rooms

        pad = 0
        for iz in range(self.size):
            for ix in range(self.size):
                if map[iz, ix]:
                    room = self.add_rect_room(
                        min_x=ix * self.room_size + pad,
                        max_x=(ix + 1) * self.room_size - pad,
                        min_z=iz * self.room_size + pad,
                        max_z=(iz + 1) * self.room_size - pad,
                        wall_tex='brick_wall',
                        floor_tex='asphalt'
                    )
                    rooms[iz][ix] = room

        # Connect rooms

        for iz in range(self.size):
            for ix in range(self.size):
                room = rooms[iz][ix]
                if room:
                    for dz, dx in [(0, 1), (1, 0)]:
                        iz1 = iz + dz
                        ix1 = ix + dx
                        if iz1 < 0 or iz1 >= self.size:
                            continue
                        if ix1 < 0 or ix1 >= self.size:
                            continue
                        neighbor = rooms[iz1][ix1]
                        if neighbor:
                            if dx == 0:
                                self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                            elif dz == 0:
                                self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

        self._map_rooms = rooms


class MazeT(MazeBase):
    """
    Like T-Maze, but Maze-T
    """

    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)

    def _gen_world(self):
        map = np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        self._gen_map_world(map)

        self.place_agent(self._map_rooms[0][4], dir=np.pi)

        goal_colors = ['red', 'blue']
        np.random.shuffle(goal_colors)
        goal_1 = self.place_entity(Box(color=goal_colors[0]), room=self._map_rooms[8][0])
        goal_2 = self.place_entity(Box(color=goal_colors[1]), room=self._map_rooms[8][8])
        self.goal_good = np.random.choice([goal_1, goal_2])
        self.goal_bad = goal_2 if self.goal_good == goal_1 else goal_1
        if self.goal_good.color == 'blue':
            self.indicator = self.place_entity(Box(color='green'), room=self._map_rooms[0][0])

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done:
            return obs, reward, done, info

        if self.near(self.goal_good):
            reward = 1.0
            done = True
        elif self.near(self.goal_bad):
            reward = -1.0
            done = True

        # Respawn instead of done
        if done:
            # Same as reset, but keep counting steps
            step_count = self.step_count
            obs = self.reset()
            self.step_count = step_count
            done = False

        return obs, reward, done, info


class MazeTEasy2(MazeT):
    def __init__(self):
        super().__init__(
            forward_step_rooms=1.0,
            turn_step=90,
            max_steps=250)


class MazeTHard2(MazeT):
    def __init__(self):
        super().__init__(
            forward_step_rooms=0.2,
            turn_step=30,
            max_steps=1500)


class MazeL(MazeBase):
    """
    Maze with loops
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_map(self):
        map = np.ones((self.size, self.size), int)

        def divide(i1: int, i2: int, j1: int, j2: int, vertical: bool, limit: int):
            if j2 - j1 > i2 - i1:
                vertical = True
            elif j2 - j1 < i2 - i1:
                vertical = False
            else:
                vertical = np.random.choice([False, True])

            if vertical:
                if j2 - j1 < 3:
                    return

                # Try put wall
                ok = False
                for _ in range(5):
                    j_wall = np.random.randint(j1 + 1, j2 - 1)
                    if (i1 > 0 and map[i1 - 1, j_wall] == 1) or (i2 < self.size and map[i2, j_wall] == 1):
                        # Invalid wall, blocks entrance
                        continue
                    ok = True
                    break
                if not ok:
                    return

                map[i1:i2, j_wall] = 0

                n_passages = 2 if (i2 - i1) == self.size else 1
                while n_passages > 0:
                    i_passage = np.random.randint(i1, i2)
                    if map[i_passage, j_wall] == 0:
                        map[i_passage, j_wall] = 1
                        n_passages -= 1

                divide(i1, i2, j1, j_wall, not vertical, limit - 1)
                divide(i1, i2, j_wall + 1, j2, not vertical, limit - 1)
            else:  # Horizontal
                if i2 - i1 < 3:
                    return

                # Try put wall
                ok = False
                for _ in range(5):
                    i_wall = np.random.randint(i1 + 1, i2 - 1)
                    if (j1 > 0 and map[i_wall, j1 - 1] == 1) or (j2 < self.size and map[i_wall, j2] == 1):
                        # Invalid wall, blocks entrance
                        continue
                    ok = True
                    break
                if not ok:
                    return

                map[i_wall, j1:j2] = 0

                n_passages = 2 if (j2 - j1) == self.size else 1
                while n_passages > 0:
                    j_passage = np.random.randint(j1, j2)
                    if map[i_wall, j_passage] == 0:
                        map[i_wall, j_passage] = 1
                        n_passages -= 1

                divide(i1, i_wall, j1, j2, not vertical, limit - 1)
                divide(i_wall + 1, i2, j1, j2, not vertical, limit - 1)

        divide(0, self.size, 0, self.size, True, 2)
        return map

    def _gen_world(self):
        map = self._gen_map()
        self._gen_map_world(map)


class MazeTriangle(MazeL):
    def _gen_world(self):
        super()._gen_world()

        def pick_room(i1, i2, j1, j2):
            while True:
                i = np.random.randint(i1, i2)
                j = np.random.randint(j1, j2)
                room = self._map_rooms[i][j]
                if room:
                    return room

        self.goal_locations = [
            pick_room(0, 9, 3, 6),
            pick_room(0, 9, 0, 3),
            pick_room(0, 9, 6, 9),
        ]

        self.place_agent(room=self.goal_locations[0])
        self.goal_index = 1
        self.goal = self.place_entity(Box(color='green'), room=self.goal_locations[1])

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.goal):
            reward = 1.0
            self.entities.remove(self.goal)
            self.goal_index = (self.goal_index + 1) % len(self.goal_locations)
            self.goal = self.place_entity(self.goal, room=self.goal_locations[self.goal_index])

        return obs, reward, done, info


class MazeTriangleN9Easy(MazeTriangle):
    def __init__(self):
        super().__init__(
            size=9,
            forward_step_rooms=1.0,
            turn_step=90,
            max_steps=500)

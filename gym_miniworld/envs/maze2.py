import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame, MeshEnt, COLORS
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

        room_texs = [
            'grass',
            'water',
            'lava',
            'wood',
            'floor_tiles_bw',
        ]
        np.random.shuffle(room_texs)

        pad = 0
        for iz in range(self.size):
            for ix in range(self.size):
                if map[iz, ix]:
                    if map[iz, ix] > 1:
                        room_id = map[iz, ix] - 2
                        wall_tex = room_texs[room_id % len(room_texs)]
                    else:
                        wall_tex = 'brick_wall'
                    room = self.add_rect_room(
                        min_x=ix * self.room_size + pad,
                        max_x=(ix + 1) * self.room_size - pad,
                        min_z=iz * self.room_size + pad,
                        max_z=(iz + 1) * self.room_size - pad,
                        wall_tex=wall_tex,
                        floor_tex='metal_grill',
                        no_ceiling=True
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
        self.teaser = self.place_entity(Box(color='yellow'), room=self._map_rooms[8][4])

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done:
            return obs, reward, done, info

        if self.teaser and self.near(self.teaser):
            self.entities.remove(self.teaser)
            self.teaser = None
            reward = 0.1

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


class MazeTEasy3(MazeT):
    def __init__(self):
        super().__init__(
            forward_step_rooms=1.0,
            turn_step=90,
            max_steps=250)


class MazeTHard3(MazeT):
    def __init__(self):
        super().__init__(
            forward_step_rooms=0.2,
            turn_step=30,
            max_steps=1500)


class MazeDMLab(MazeBase):
    """
    Maze with layout using DMLab maze generator
    """

    def __init__(self,
                 size=9,
                 max_rooms=4,
                 room_min_size=3,
                 room_max_size=5,
                 **kwargs):
        self.size = size
        self.max_rooms = max_rooms
        self.room_min_size = room_min_size
        self.room_max_size = room_max_size
        super().__init__(size=size, **kwargs)

    def _gen_world(self, room_object_count=0):
        from dmlab_maze_generator import create_random_maze
        maze = create_random_maze(width=self.size + 2,
                                  height=self.size + 2,
                                  max_rooms=self.max_rooms,
                                  room_max_size=self.room_max_size,
                                  room_min_size=self.room_min_size,
                                  room_object_count=room_object_count)
        maze = [row[1:-1] for row in maze.split('\n')[1:-1]]  # remove outer walls
        maze = np.array([np.array(list(row)) for row in maze])  # to np.array of chars
        map = (maze != '*').astype(int)

        # Mark rooms with different numbers
        #  0 - wall
        #  1 - corridor
        #  2, 3, ... - different rooms
        n_rooms = 0
        for i in range(map.shape[0] - 1):
            for j in range(map.shape[1] - 1):
                if map[i, j] and map[i + 1, j] and map[i, j + 1] and map[i + 1, j + 1]:
                    # Belongs to a room
                    if i > 0 and map[i - 1, j] >= 2:
                        i_room = map[i - 1, j]  # same room
                    elif j > 0 and map[i, j - 1] >= 2:
                        i_room = map[i, j - 1]  # same room
                    else:
                        i_room = 2 + n_rooms  # new room
                        n_rooms += 1
                    map[i, j] = i_room
                    map[i + 1, j] = i_room
                    map[i, j + 1] = i_room
                    map[i + 1, j + 1] = i_room

        self._gen_map_world(map)
        self._maze = maze


class ScavengerHunt(MazeDMLab):
    MAX_GOALS_PER_ROOM = 1

    def __init__(self,
                 n_goals,
                 with_decor=True,
                 with_crumbs=False, 
                 **kwargs):
        assert n_goals <= len(COLORS)
        self.n_goals = n_goals
        self.with_decor = with_decor
        self.with_crumbs = with_crumbs
        super().__init__(**kwargs)

    def _gen_world(self):
        k = self.size * self.size // 80
        DECORATIONS = [
            (k, 'tree_pine', 1.5, 3.0),
            (k, 'office_chair', 1.0, 2.0),
            (k, 'duckie', 0.5, 1.0),
            (k if not self.with_crumbs else 0, 'cone', 0.5, 1.0),  # if self.crumbs, then use cones as crumbs instead
            (k, 'medkit', 0.4, 0.8),
        ]

        # Generate maze

        super()._gen_world(room_object_count=self.MAX_GOALS_PER_ROOM)
        decor_locations = [self._map_rooms[i][j] for i, j in zip(*np.where(self._maze == ' '))]  # potential locations for decorations
        goal_locations = [self._map_rooms[i][j] for i, j in zip(*np.where(self._maze == 'O'))]  # potential locations for goals

        # Goal objects

        self.goals = []
        for color in list(COLORS.keys())[:self.n_goals]:
            if not goal_locations:
                break
            room = np.random.choice(goal_locations)
            goal_locations.remove(room)
            obj = Box(color=color)
            obj.solid = False
            goal = self.place_entity(obj, room=room)
            self.goals.append(goal)

        # Decoration objects

        if self.with_decor:
            for n, mesh, min_height, max_height in DECORATIONS:
                for _ in range(n):
                    room = np.random.choice(decor_locations)
                    decor_locations.remove(room)
                    self.place_entity(
                        MeshEnt(mesh_name=mesh,
                                height=min_height + np.random.rand() * (max_height - min_height),
                                solid=False),
                        room=room)

        # Fill crumbs objects, for encouraging exploration

        self.crumbs = []
        if self.with_crumbs:
            while len(decor_locations) > 0:
                room = np.random.choice(decor_locations)
                decor_locations.remove(room)
                o = self.place_entity(MeshEnt(mesh_name='cone', height=0.5, solid=False, static=False), room=room)
                self.crumbs.append(o)

        # Agent

        self.place_agent()
        self.goal = np.random.choice(self.goals)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Crumbs

        for o in list(self.crumbs):
            if self.near(o):
                reward += 0.1
                self.entities.remove(o)
                self.crumbs.remove(o)

        # Goal

        if self.near(self.goal):
            reward += 1.0
            self.goal = np.random.choice(list(set(self.goals) - set([self.goal])))
            obs = self.render_obs()  # re-render with new goal

        self.reward_sum += reward
        return obs, reward, done, info

    def render_obs(self, frame_buffer=None):
        obs = super().render_obs(frame_buffer)
        B = 2
        obs[:, :B] = self.goal.color_vec * 255
        obs[:, -B:] = self.goal.color_vec * 255
        obs[:B, :] = self.goal.color_vec * 255
        obs[-B:, :] = self.goal.color_vec * 255
        return obs


class ScavengerHuntSmall(ScavengerHunt):
    def __init__(self):
        # Maze based on DMLab30-explore_goal_locations_small
        # {
        #     mazeHeight = 11,  # with outer walls
        #     mazeWidth = 11,
        #     roomCount = 4,
        #     roomMaxSize = 5,
        #     roomMinSize = 3,
        # }
        super().__init__(
            size=9,  # without outer walls
            n_goals=3,
            max_rooms=4,
            room_min_size=3,
            room_max_size=5,
            forward_step_rooms=0.33,
            turn_step=90 / 4,
            max_steps=1000)


class ScavengerHuntMedium(ScavengerHunt):
    def __init__(self):
        super().__init__(
            size=11,
            n_goals=4,
            max_rooms=6,
            room_min_size=3,
            room_max_size=5,
            forward_step_rooms=0.33,
            turn_step=90 / 4,
            max_steps=2000)


class ScavengerHuntMediumNodec(ScavengerHunt):
    def __init__(self):
        super().__init__(
            with_decor=False, # Nodec
            size=11,
            n_goals=4,
            max_rooms=6,
            room_min_size=3,
            room_max_size=5,
            forward_step_rooms=0.33,
            turn_step=90 / 4,
            max_steps=2000)

class ScavengerHuntLargish(ScavengerHunt):
    def __init__(self):
        super().__init__(
            size=13,
            n_goals=5,
            max_rooms=6,
            room_min_size=3,
            room_max_size=5,
            forward_step_rooms=0.33,
            turn_step=90 / 4,
            max_steps=3000)

class ScavengerHuntLarge(ScavengerHunt):
    def __init__(self):
        # Maze based on DMLab30-explore_goal_locations_large
        # {
        #     mazeHeight = 17,  # with outer walls
        #     mazeWidth = 17,
        #     roomCount = 9,
        #     roomMaxSize = 3,
        #     roomMaxSize = 3,
        # }
        super().__init__(
            size=15,  # without outer walls
            n_goals=6,
            max_rooms=9,
            room_min_size=3,
            room_max_size=3,
            forward_step_rooms=0.33,
            turn_step=90 / 4,
            max_steps=4000)

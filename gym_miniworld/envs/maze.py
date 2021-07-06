import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame
from ..params import DEFAULT_PARAMS

class Maze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=8,
        num_cols=8,
        room_size=3,
        gap_size=0.25,
        max_episode_steps=None,
        agent_start_topleft=False,
        no_goal=False,
        n_apples=0,
        no_texture=False,
        **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = gap_size
        self.agent_start_topleft = agent_start_topleft
        self.no_goal = no_goal
        self.n_apples = n_apples
        self.no_texture = no_texture

        super().__init__(
            max_episode_steps = max_episode_steps or num_rows * num_cols * 24,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='white' if self.no_texture else 'brick_wall',
                    floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = self.rand.subset([(0,1), (0,-1), (-1,0), (1,0)], 4)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        if self.agent_start_topleft:
            pos = self.room_size/2
            self.place_agent(dir=0, min_x=pos, max_x=pos, min_z=pos, max_z=pos)
        else:
            self.place_agent()

        if not self.no_goal:
            self.goal_box = self.place_entity(Box(color='red'))
        else:
            self.goal_box = None

        self.apples = []
        for _ in range(self.n_apples):
            self.apples.append(self.place_entity(Box(color='green')))

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.goal_box and self.near(self.goal_box):
            reward += self._reward()
            done = True

        for apple in self.apples[:]:
            if self.near(apple):
                self.entities.remove(apple)
                self.apples.remove(apple)
                reward += 1.0  # Reward for each
                if self.n_apples > 0 and len(self.apples) == 0:
                    done = True  # All collected

        return obs, reward, done, info

class MazeS2(Maze):
    def __init__(self):
        super().__init__(num_rows=2, num_cols=2)

class MazeS3(Maze):
    def __init__(self):
        super().__init__(num_rows=3, num_cols=3)


class MazeS5A4(Maze):
    def __init__(self, size=5, max_steps=2000):
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 1)
        params.set('turn_step', 25)
        super().__init__(
            num_rows=size, 
            num_cols=size, 
            params=params, 
            max_episode_steps=max_steps,
            room_size=3,
            gap_size=3,
            no_goal=True,
            n_apples=4,
            obs_width=64,
            obs_height=64,
        )

class MazeS5A4W(Maze):
    def __init__(self, size=5, max_steps=2000):
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 1)
        params.set('turn_step', 25)
        super().__init__(
            num_rows=size, 
            num_cols=size, 
            params=params, 
            max_episode_steps=max_steps,
            room_size=3,
            gap_size=3,
            no_goal=True,
            n_apples=4,
            obs_width=64,
            obs_height=64,
            no_texture=True,
        )


class MazeS5N(Maze):
    def __init__(self, size=5, max_steps=2000):
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 1)
        params.set('turn_step', 25)
        super().__init__(
            num_rows=size, 
            num_cols=size, 
            params=params, 
            max_episode_steps=max_steps,
            room_size=3,
            gap_size=3,
            # agent_start _topleft=True,
            no_goal=True,
            obs_width=64,
            obs_height=64,
        )

class MazeS5GridN(Maze):
    def __init__(self, size=5, max_steps=500):
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 2)
        params.set('turn_step', 90)
        super().__init__(
            num_rows=size, 
            num_cols=size, 
            params=params, 
            max_episode_steps=max_steps,
            room_size=2,
            gap_size=2,
            agent_start_topleft=True,
            no_goal=True,
            obs_width=64,
            obs_height=64,
            )

class MazeS5GridA4(Maze):
    def __init__(self, size=5, max_steps=500):
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 3)
        params.set('turn_step', 90)
        super().__init__(
            num_rows=size, 
            num_cols=size, 
            params=params, 
            max_episode_steps=max_steps,
            room_size=3,
            gap_size=3,
            no_goal=True,
            n_apples=4,
            obs_width=64,
            obs_height=64,
            )

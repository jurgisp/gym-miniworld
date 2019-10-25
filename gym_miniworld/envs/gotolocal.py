import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room, COLOR_NAMES
from ..entity import Box

class GoToLocal(MiniWorldEnv):
    """
    Single-room environment where a red box must be placed next
    to a yellow box.
    """

    def __init__(self, size=10, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=125,
            **kwargs
        )

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        for color in COLOR_NAMES:
            box = Box(color=color, size=self.rand.float(0.80, 0.85))
            self.place_entity(box)

            if box.color == "red":
                self.red_box = box
            elif box.color == "yellow":
                self.yellow_box = box

        # Place the agent a random distance away from the goal
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.red_box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

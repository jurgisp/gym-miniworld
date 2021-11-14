#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
from gym_miniworld.wrappers import *

FPS = 6

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--map_wrapper', action='store_true', help='use map wrapper')
args = parser.parse_args()

env = gym.make(args.env_name)
if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

if args.map_wrapper:
    env = DictWrapper(env)
    env = MapWrapper(env)
    # env = PixelMapWrapper(env)
    env = AgentPosWrapper(env)

view_mode = 'top' if args.top_view else 'agent'

env.reset()

# Create the display window
env.render('pyglet', view=view_mode)


def step(action):
    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    obs, reward, done, info = env.step(action)
    if isinstance(obs, dict):
        # print({k: v.shape for k, v in obs.items()})
        # print(obs['agent_pos'], obs['agent_dir'])
        # print(obs['map_agent'].T)
        pass

    if reward != 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        env.reset()

    env.render('pyglet', view=view_mode)

def update(*args):
    if keys[key.UP] or keys[key.W]:
        step(env.actions.move_forward)
    elif keys[key.DOWN] or keys[key.S]:
        step(env.actions.move_back)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render('pyglet', view=view_mode)
        return

    if symbol == key.ESCAPE:
        env.close()
        pyglet.app.exit()

    # elif symbol == key.UP or symbol == key.W:
    #     step(env.actions.move_forward)
    # elif symbol == key.DOWN or symbol == key.S:
    #     step(env.actions.move_back)

    elif symbol == key.LEFT or symbol == key.A:
        step(env.actions.turn_left)
    elif symbol == key.RIGHT or symbol == key.D:
        step(env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    elif symbol == key.ENTER:
        step(env.actions.done)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()


# Enter main event loop
keys = key.KeyStateHandler()
env.window.push_handlers(keys)
pyglet.clock.schedule_interval(update, 1/FPS)
pyglet.app.run()


env.close()

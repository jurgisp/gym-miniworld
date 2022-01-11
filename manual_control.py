#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

from collections import defaultdict
import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
from gym_miniworld.wrappers import *
from recording import *

FPS = 6

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--map_wrapper', action='store_true', help='use map wrapper')
parser.add_argument('--record', action='store_true', help='record episodes')
args = parser.parse_args()

env = gym.make(args.env_name)
if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

if args.map_wrapper or args.record:
    env = DictWrapper(env)
    env = MapWrapper(env)
    env = AgentPosWrapper(env)
    env = GoalPosWrapper(env)
    env = GoalVisibleWrapper(env)
    env = GoalVisAgeWrapper(env)

if args.record:
    env = ActionRewardResetWrapper(env)
    env = CollectWrapper(env)
    env = SaveNpzWrapper(env, './log')

view_mode = 'top' if args.top_view else 'agent'

obs = env.reset()
# print({k: v.shape for k, v in obs.items()})

# Create the display window
env.render('pyglet', view=view_mode)


def step(action):
    if action >= env.action_space.n:
        return

    obs, reward, done, info = env.step(action)

    if reward != 0:
        print(f'{env.step_count}/{env.max_episode_steps} action={action} reward={reward:.1f}')
    # print(obs['agent_pos'], obs['agent_dir'])
    # print(obs['map_agent'].T)
    # print(obs['goals_visage'])

    if done:
        print('done!')
        env.reset()

    env.render('pyglet', view=view_mode)

def handle_keys(*args):
    keys_active = defaultdict(bool)
    now = time.time()

    for k in keys_pressed:
        if keys_pressed[k] and keys_cooldown.get(k, 0) < now:
            keys_active[k] = True
            keys_cooldown[k] = now + 1 / FPS

    if keys_active[key.LEFT] or keys_active[key.A]:
        step(env.actions.turn_left)
    
    elif keys_active[key.RIGHT] or keys_active[key.D]:
        step(env.actions.turn_right)

    elif keys_active[key.UP] or keys_active[key.W]:
        step(env.actions.move_forward)

    elif keys_active[key.DOWN] or keys_active[key.S]:
        step(env.actions.move_back)
    


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        obs = env.reset()
        # print(obs['goals_visage'])
        env.render('pyglet', view=view_mode)
        return

    if symbol == key.ESCAPE:
        env.close()
        pyglet.app.exit()

    # elif symbol == key.UP or symbol == key.W:
    #     step(env.actions.move_forward)
    # elif symbol == key.DOWN or symbol == key.S:
    #     step(env.actions.move_back)
    # elif symbol == key.LEFT or symbol == key.A:
    #     step(env.actions.turn_left)
    # elif symbol == key.RIGHT or symbol == key.D:
    #     step(env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    # elif symbol == key.ENTER:
    #     step(env.actions.done)

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
keys_pressed = key.KeyStateHandler()
keys_cooldown = {}
env.window.push_handlers(keys_pressed)
pyglet.clock.schedule_interval(handle_keys, 1/30)
pyglet.app.run()


env.close()

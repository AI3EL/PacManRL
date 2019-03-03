# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:33:12 2019

@author: jacqu
"""

import numpy as np
import random
import pygame
from pygame.locals import *

from env_utils import get_direction, ghost_threats

WINDOW_SHAPE = 500, 500
WALL_COLOR = (0, 0, 0)
VOID_COLOR = (255,255,255)
SB_COLOR = (255,255,0)
BB_COLOR = SB_COLOR
GHOST_COLOR = (255, 0, 0)
PACMAN_COLOR = (0, 0, 255)
SUPER_PACMAN_COLOR = (0, 255, 0)
SUPER_TIMEOUT = 40
MOVE_TIME = 150

# if setting state to "vector10", new and more simple state representation:
# Observation is a vector of shape 10:
# First four features indicate presence of walls around
# Fifth is direction of best target
# Sixth to nineth indicate presence of ghost threat in direction
# last one indicates whether pacman is trapped (see paper for why it is important...)


# 0 = nothing, 1 = wall, 2 = small ball, 3 = big ball
class PacManEnv:
    directions = ['left', 'down', 'right', 'up']
    objects = ['void', 'wall', 'sb', 'bb']
    ghost_change_freq = 0.2

    @staticmethod
    def load_map(map_name):
        with open(map_name, 'r') as f:
            w, h = map(int, f.readline().split(','))
            res = np.zeros((w, h), int)
            for i in range(h):
                res[i] = list(map(int, list(f.readline())[:-1]))
            return res

    def __init__(self, map_name, pac_position, ghost_positions, ghost_directions, state="usual", time_out=None, death_cost=0):
        """ parameter state replaces bool low_dim and specifies the state representation we choose.
        Is string type, and can be "usual", "low_dim" or "vector10". More can be added in the future"""

        self.map = PacManEnv.load_map(map_name)
        self.pac_position = pac_position
        self.ghost_positions = ghost_positions
        self.ghost_directions = ghost_directions
        self.super_mode = False
        self.super_timeout = 0
        self.state_rpz = state
        self.time = 0
        self.time_out = time_out
        self.death_cost = death_cost

        # Setting observation dimension: depends on which state representation we choose
        if (state=="usual"):
            self.observation_dim = self.map.shape[0]*self.map.shape[1] + len(ghost_positions)*2 + 2
        elif (state=="low_dim"):
            self.observation_dim = len(ghost_positions)*2 + 2
        elif(state=="vector10"):
            self.observation_dim = 10
        self.action_size = 4

        # To be able to env.reset()
        self.init_map = self.map.copy()
        self.init_pac_position = pac_position
        self.init_ghost_positions = ghost_positions.copy()
        self.init_ghost_directions = ghost_directions.copy()

        pygame.init()
        self.window = pygame.display.set_mode(WINDOW_SHAPE)

    def get_qtable(self):
        return np.ones((self.map.shape[0], self.map.shape[1], len(self.objects), len(self.directions)))

    # Assumes map is well made
    def move(self, pos, direction):
        if PacManEnv.directions[direction] == 'right':
            return pos[0], (pos[1] + 1) % self.map.shape[1]
        elif PacManEnv.directions[direction] == 'left':
            return pos[0], (pos[1] - 1) % self.map.shape[1]
        elif PacManEnv.directions[direction] == 'up':
            return (pos[0] - 1) % self.map.shape[0], pos[1]
        elif PacManEnv.directions[direction] == 'down':
            return (pos[0] + 1) % self.map.shape[0], pos[1]
        else:
            raise ValueError('Incorrect direction', direction)

    def non_block_directions(self, pos):
        """0 is left, 1 is down, 2 is right, 3 is up"""
        res = []
        if self.map[pos[0], (pos[1] - 1) % self.map.shape[1]] != 1: # left direction
            res.append(0)
        if self.map[(pos[0] + 1) % self.map.shape[0], pos[1]] != 1: # down direction
            res.append(1)
        if self.map[pos[0], (pos[1] + 1) % self.map.shape[1]] != 1: # right direction
            res.append(2)
        if self.map[(pos[0] - 1) % self.map.shape[0], pos[1]] != 1: # up direction
            res.append(3)
        return res

    # Just checks there is no wall
    def move_if_valid(self, pos, direction):
        next_position = self.move(pos, direction)
        if self.map[next_position] != 1:
            return next_position
        return pos

    def get_observation(self):
        """ returns the observation, of dimension self.observation_dim"""
        if (self.state_rpz=="low_dim"):
            return self.pac_position, self.ghost_positions

        elif(self.state_rpz=="usual"):
            observation = self.map.copy()
            if self.super_mode:
                observation[self.pac_position] = 5
            else:
                observation[self.pac_position] = 4
            for ghos_pos in self.ghost_positions:
                observation[ghos_pos] = 6
            return observation

        elif(self.state_rpz=="vector10"):
            # build and return observation vector of length 10
            s=np.zeros(10)
            # we give the information about non blocking directions
            non_blocking_dir=self.non_block_directions(self.pac_position)
            for i in range(4):
                if(i in non_blocking_dir):
                    s[i]=1

            # now we compute the direction of closest target
            best_direction=get_direction(self.map, self.pac_position, self.ghost_positions)
            s[4]=best_direction

            # now we evaluate the presence of ghost threat in each direction (only ones with no wall...)
            s[5:9]=ghost_threats(self.map, self.pac_position, self.ghost_positions, self.ghost_directions)

            # now we evaluate whether pac is blocked : this happens if all 4 directions are bad options.
            problems=np.concatenate((s[1:4],s[5:9]))
            if(np.sum(problems)==4): # a problem (wall or threat) is detected in each direction
                s[9]=1

            # return the observation vector
            return s

    # Returns : observation, reward, done, info
    def step(self, action):
        reward = 0

        # Decrease super_mode timeout
        if self.super_mode:
            self.super_timeout -= 1
            if self.super_timeout <= 0:
                self.super_mode = False

        # Move pacman
        self.pac_position = self.move_if_valid(self.pac_position, action)

        # Has to check before and after ghost position change
        for i in range(len(self.ghost_positions)):
            if self.ghost_positions[i] == self.pac_position:
                if self.super_mode:
                    self.ghost_positions[i] = (-1, -1)
                    self.ghost_directions[i] = -1
                    reward += 10
                else:
                    observation = self.get_observation()
                    return observation, -self.death_cost, True, 'Failure'

            # Move if not dead
            if self.ghost_positions[i] != (-1, -1):
                if self.super_mode:
                    if self.time % 4 == 0:
                        self.ghost_directions[i] = random.sample(self.non_block_directions(self.ghost_positions[i]), 1)[0]
                        self.ghost_positions[i] = self.move_if_valid(self.ghost_positions[i], self.ghost_directions[i])
                else:
                    if self.time % 2:
                        self.ghost_directions[i] = random.sample(self.non_block_directions(self.ghost_positions[i]), 1)[0]
                        self.ghost_positions[i] = self.move_if_valid(self.ghost_positions[i], self.ghost_directions[i])

            if self.ghost_positions[i] == self.pac_position:
                if self.super_mode:
                    self.ghost_positions[i] = (-1, -1)
                    self.ghost_directions[i] = -1
                    reward += 10
                else:
                    observation = self.get_observation()
                    return observation, -self.death_cost, True, 'Failure'

        if self.map[self.pac_position] == 2:
            self.map[self.pac_position] = 0
            reward += 1

        elif self.map[self.pac_position] == 3:
            self.map[self.pac_position] = 0
            reward += 5
            self.super_mode = True
            self.super_timeout = SUPER_TIMEOUT

        if self.time >= self.time_out:
            return self.get_observation(), reward, True, 'Failure'

        else:
            self.time += 1
            return self.get_observation(), reward, False, ''

    def render(self):
        self.window.fill(VOID_COLOR)
        px = (WINDOW_SHAPE[0] / self.map.shape[0], WINDOW_SHAPE[1] / self.map.shape[1])
        for i, j in np.ndindex(self.map.shape):
            color = VOID_COLOR
            size = px
            off_set = (0, 0)
            if self.map[i, j] == 1:
                color = WALL_COLOR
            elif self.map[i, j] == 2:
                off_set = (px[0]*0.35, px[1]*0.35)
                size = (px[0]*0.3, px[1]*0.3)
                color = SB_COLOR
            elif self.map[i, j] == 3:
                off_set = (px[0] * 0.2, px[1] * 0.2)
                size = (px[0] * 0.6, px[1] * 0.6)
                color = BB_COLOR
            pygame.draw.rect(self.window, color, Rect([px[0] * i + off_set[0], px[1] * j + off_set[1]], size))
        for pos in self.ghost_positions:
            pygame.draw.rect(self.window, GHOST_COLOR, Rect([px[0] * pos[0], px[1] * pos[1]], px))
        if self.super_mode:
            pygame.draw.rect(self.window, SUPER_PACMAN_COLOR,
                             Rect([px[0] * self.pac_position[0], px[1] * self.pac_position[1]], px))
        else:
            pygame.draw.rect(self.window, PACMAN_COLOR,
                             Rect([px[0] * self.pac_position[0], px[1] * self.pac_position[1]], px))
        pygame.display.update()
        pygame.time.wait(MOVE_TIME)

    def reset(self):
        self.map = self.init_map.copy()
        self.pac_position = self.init_pac_position
        self.ghost_positions = self.init_ghost_positions.copy()
        self.ghost_directions = self.init_ghost_directions.copy()
        self.time = 0
        self.super_mode = False
        return self.get_observation()
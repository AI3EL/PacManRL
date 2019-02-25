import numpy as np
import random
import pygame
from pygame.locals import *

WINDOW_SHAPE = 500, 500
WALL_COLOR = (0, 0, 0)
VOID_COLOR = (255,255,255)
SB_COLOR = (255,255,0)
GHOST_COLOR = (255, 0, 0)
PACMAN_COLOR = (0, 0, 255)
MOVE_TIME = 300


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

    def __init__(self, map_name, pac_position, ghost_positions, ghost_directions):
        self.map = PacManEnv.load_map(map_name)
        self.pac_position = pac_position
        self.ghost_positions = ghost_positions
        self.ghost_directions = ghost_directions
        self.observation_dim = self.map.shape[0]*self.map.shape[1] + len(ghost_positions)*2 + 2
        if ghost_positions:
            self.observation_shape = (self.map.shape[0], self.map.shape[1], len(ghost_positions)*2, 2)
        else:
            self.observation_shape = (self.map.shape[0], self.map.shape[1], 2)
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

    # Just checks there is no wall
    def move_if_valid(self, pos, direction):
        next_position = self.move(pos, direction)
        if self.map[next_position] != 1:
            return next_position
        return pos

    # Returns : observation, reward, done, info
    def step(self, action):
        reward = 0

        self.pac_position = self.move_if_valid(self.pac_position, action)

        for i in range(len(self.ghost_positions)):
            if random.random() < PacManEnv.ghost_change_freq:  # Decide to change direction of ghost
                self.ghost_directions[i] = random.randint(0, 3)
            self.ghost_positions[i] = self.move_if_valid(self.ghost_positions[i], self.ghost_directions[i])
            if self.ghost_positions[i] == self.pac_position:
                observation = (self.map, self.pac_position, self.ghost_positions)
                return observation, reward, True, 'Failure'

        if self.map[self.pac_position] == 2:
            self.map[self.pac_position] = 0
            reward = 1

        observation = (self.map, self.pac_position, self.ghost_positions)
        return observation, reward, False, ''

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
                off_set = (px[0] / 4, px[1] / 4)
                size = (px[0] / 2, px[1] / 2)
                color = SB_COLOR
            elif self.map[i, j] == 3:
                color = SB_COLOR
            pygame.draw.rect(self.window, color, Rect([px[0] * i + off_set[0], px[1] * j + off_set[1]], size))
        for pos in self.ghost_positions:
            pygame.draw.rect(self.window, GHOST_COLOR, Rect([px[0] * pos[0], px[1] * pos[1]], px))
        pygame.draw.rect(self.window, PACMAN_COLOR, Rect([px[0] * self.pac_position[0], px[1] * self.pac_position[1]], px))
        pygame.display.update()
        pygame.time.wait(MOVE_TIME)

    def reset(self):
        self.map = self.init_map.copy()
        self.pac_position = self.init_pac_position
        self.ghost_positions = self.init_ghost_positions.copy()
        self.ghost_directions = self.init_ghost_directions.copy()
        observation = (self.map, self.pac_position, self.ghost_positions)
        return observation
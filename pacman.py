import numpy as np
import random


# 0 = nothing, 1 = wall, 2 = small ball, 3 = big ball
class PacManEnv:
    directions = ['left', 'down', 'right', 'up']
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

        # To be able to env.reset()
        self.init_map = self.map.copy()
        self.init_pac_position = pac_position
        self.init_ghost_positions = ghost_positions.copy()
        self.init_ghost_directions = ghost_directions.copy()

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
                return None, reward, True, 'Failure'

        if self.map[self.pac_position] == 2:
            self.map[self.pac_position] = 0
            reward = 1

        observation = (self.map, self.pac_position, self.ghost_positions)
        return observation, reward, False, ''

    def render(self):
        rendering = np.zeros(self.map.shape, str)
        for i, j in np.ndindex(self.map.shape):
            if self.map[i,j] == 0:
                rendering[i,j] = ' '
            elif self.map[i,j] == 1:
                rendering[i,j] = 'X'
            elif self.map[i,j] == 2:
                rendering[i,j] = 'o'
        rendering[self.pac_position] = 'P'
        for ghost_position in self.ghost_positions:
            rendering[ghost_position] = 'G'
        print(rendering)

    def reset(self):
        self.map = self.init_map
        self.pac_position = self.init_pac_position
        self.ghost_positions = self.init_ghost_positions
        self.ghost_directions = self.init_ghost_directions


env = PacManEnv('map1.txt', (4,6), [(7,1)], [2])

env.render()
n_step = 100
n_episode = 20
for i_episode in range(n_episode):
    env.reset()
    score = 0
    for i_step in range(n_step):
        observation, reward, done, info = env.step(random.randint(0, 3))
        score += reward
        # env.render()
        if done:
            print('Episode ended in {0} steps, score is {1}'.format(i_step, score))
            env.render()
            break

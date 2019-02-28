import itertools
import random
import numpy as np


# Uses algorithm REINFORCE with 2 sigmoids reprensenting the policy
# Poor results : not able to get small balls even when no ghost

def sigmoid(theta, x):
    assert len(theta) == len(x)
    return 1 / (1 + np.exp(-sum([theta[i]*x[i] for i in range(len(theta))])))


def apply_policy(theta1, theta2, s):
    a = random.random() <= sigmoid(theta1, s)
    b = random.random() <= sigmoid(theta2, s)
    if a and b:
        return 0
    elif a and not b:
        return 1
    elif b and not a:
        return 2
    else:
        return 3


def policy_gradient(states, actions, rewards, theta):
    assert len(states) == len(actions) == len(rewards)
    n = len(states)
    state_dim = len(states[0])
    gradient = [0]*state_dim
    cur_reward = sum(rewards)
    for i in range(n):
        factor = cur_reward
        if actions[i]:
            factor *= 1-sigmoid(theta, states[i])
        else:
            factor *= -sigmoid(theta, states[i])
        for j in range(state_dim):
            gradient[j] += factor*states[i][j]
        cur_reward -= rewards[i]
    return gradient


def update_theta(theta, alpha, states, actions, rewards):
    state_dim = len(states[0])
    gradient = policy_gradient(states, actions, rewards, theta)
    return [theta[i] + alpha * gradient[i] for i in range(state_dim)]


def update_alpha(start_alpha, i):
    return start_alpha /np.sqrt(1+i)

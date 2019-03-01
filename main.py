from environment import PacManEnv
from reinforce import *
from DQN import DQN
from utils import vectorize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def use_reinforce(env, n_episode, n_step, start_alpha, info_times = 20):
    print('Start REINFORCE with', n_episode, n_step, start_alpha, info_times)

    theta1, theta2 = [0. for i in range(env.observation_dim)], [0. for i in range(env.observation_dim)]
    alpha = start_alpha
    average_score = 0

    for i_episode in range(n_episode):
        observation = vectorize(env.reset())
        score = 0
        actions = []
        states = []
        rewards = []

        for i_step in range(n_step):
            prev_observation = observation.copy()
            action = apply_policy(theta1, theta2, observation)
            if i_episode == n_episode - 1:
                env.render()
            observation, reward, done, info = env.step(action)
            observation = vectorize(observation)
            states.append(prev_observation)
            actions.append(action)
            rewards.append(reward)

            score += reward
            # env.render()
            if done:
                break

        alpha = update_alpha(start_alpha, i_episode)
        average_score += score

        if i_episode and i_episode % (n_episode / info_times) == 0:
            average_score /= (n_episode / info_times)
            print('Progression : {0} Average Score : {1}, Alpha : {2}'.format(i_episode / n_episode, average_score,
                                                                              alpha))

        # print('Episode ended in {0} steps, score is {1}'.format(survival_time, score))

        # print(theta1)
        # alpha = update_alpha(score)
        theta1 = update_theta(theta1, alpha, states, actions, rewards)
        theta2 = update_theta(theta2, alpha, states, actions, rewards)


def plot_logs(logs, log_freq):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate', 'violet']
    legend = []
    for i in range(len(logs)):
        absc = [1./log_freq * i for i in range(log_freq)]
        ords = logs[i]
        plt.plot(absc, ords, color=colors[i])
        legend.append(mpatches.Patch(color=colors[i], label='Epoch {}'.format(i)))
    plt.legend(handles=legend)
    plt.show()


def save_logs_plot(logs, filename):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate', 'violet']
    legend = []
    n_files = 0
    for i in range(len(logs)):
        log_freq = len(logs[i])
        absc = [1. / log_freq * i for i in range(log_freq)]
        ords = logs[i]
        plt.plot(absc, ords, color=colors[i])
        legend.append(mpatches.Patch(color=colors[i], label='Epoch {}'.format(i)))
        if i and not(i % 4):
            plt.legend(handles=legend)
            plt.savefig(filename + '{}.png'.format(n_files))
            legend = []
            plt.clf()
            n_files += 1
    plt.legend(handles=legend)
    plt.savefig(filename + '{}.png'.format(n_files))


env = PacManEnv('map1.txt', (4, 6), [(7, 1)], [2], "usual", 50)
# env = PacManEnv('map1.txt', (4, 6), [], [], "usual", 50)
# env = PacManEnv('map2.txt', (3, 3), [], [], False, 8)
# env = PacManEnv('map2.txt', (3, 3), [(1, 1)], [2], False, 20, 20)

dqn = DQN(env, 500, [128])
eps_schedule = [(1., 0.1, 0.8)] + [(0.35, 0.1, 0.1)]*3
logs = dqn.train(eps_schedule, 0.99, 10000, 16, 100, 50, 50)
save_logs_plot(logs, 'NN=128,T=10000,D=500,C=100,batch=16')

dqn = DQN(env, 500, [128])
eps_schedule = [(1., 0.1, 0.8)] + [(0.35, 0.1, 0.1)]*3
logs = dqn.train(eps_schedule, 0.99, 10000, 16, 500, 50, 50)
save_logs_plot(logs, 'NN=128,T=10000,D=500,C=100,batch=16')

dqn = DQN(env, 500, [128])
eps_schedule = [(1., 0.1, 0.8)] + [(0.35, 0.1, 0.1)]*3
logs = dqn.train(eps_schedule, 0.99, 10000, 16, 1000, 50, 50)
save_logs_plot(logs, 'NN=128,T=10000,D=500,C=100,batch=16')

#dqn.observe(5)
# dqn.summary()


# print("Final qtable : ")
# for k,v in dqn.get_q_table().items():
#     print(k,v)

# n_episode = 500
# n_step = 20
# info_times = 20
# start_alpha = 0.1 / env.observation_dim
# use_reinforce(env, n_episode, n_step, start_alpha)

from environment import PacManEnv
from reinforce import *
from DQN import DQN


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


# env = PacManEnv('map1.txt', (4, 6), [(7,1)], [2])
# env = PacManEnv('map1.txt', (4, 6), [], [])
env = PacManEnv('map2.txt', (3, 3), [], [])

dqn = DQN(env, 50)
initial_q_table = dqn.get_q_table()
dqn.train(0.99, 0.7, 1000, 1, 10, 10)
for k, v in dqn.get_q_table().items():
    print(k, initial_q_table[k] - v)
# dqn.observe()

# n_episode = 500
# n_step = 20
# info_times = 20
# start_alpha = 0.01 / env.observation_dim
# use_reinforce(env, n_episode, n_step, start_alpha)

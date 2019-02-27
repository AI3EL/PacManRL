import numpy as np
import random
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense

# TODO : vérifier si l'algo est bien implémenté
# TODO : vérifier que ca marche avec les fantomes
# TODO : printer des trucs / voir l'évolution du NN pour essayer de comprendre pourquoi ca marche pas


class DQN:
    def __init__(self, env, buffer_size):
        self.env = env
        self.QNN = self.build_QNN(env.observation_dim)  # Q network
        self.TNN = self.build_QNN(env.observation_dim)  # Target network
        self.D = []  # Pool containing experiences
        self.D_size = buffer_size

        # Variables used to recall the environment state when filling D
        self.cur_observation = None
        self.time_alive = 0

    @staticmethod
    def build_QNN(input_dim):
        hidden_nodes = 32
        model = Sequential()
        model.add(Dense(hidden_nodes, input_dim=input_dim, activation='relu', name='hidden1'))
        model.add(Dense(4, activation='softmax', name='final'))
        model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.RMSprop(),
            metrics=['accuracy']
        )
        return model

    def get_mini_batch(self, size):
        return random.sample(self.D, size)

    def fill_D(self, eps, n, max_t):
        count = 0
        while len(self.D) < self.D_size and count < n:
            # Choosing and performing action
            if random.random() < eps:
                action = random.randint(0, self.env.action_size - 1)
            else:
                action = np.argmax(self.QNN.predict(to_array(self.cur_observation), batch_size=1))
            next_observation, reward, done, info = self.env.step(action)
            count += 1

            if done or self.time_alive >= max_t:
                self.D.append((self.cur_observation, action, reward, None))
                self.cur_observation = self.env.reset()
                self.time_alive = 0
            else:
                self.D.append((self.cur_observation, action, reward, next_observation))
                self.cur_observation = next_observation
            self.time_alive += 1

    def get_training_set(self, batch, gamma):
        y_train = []
        for i in range(len(batch)):
            tmp = self.QNN.predict(to_array(batch[i][0]))[0]  # NOT optimal but simpler with keras
            if batch[i][3] is None:
                tmp[batch[i][1]] = batch[i][2]
            else:
                tmp[batch[i][1]] = batch[i][2] + gamma * max(self.TNN.predict(to_array(batch[i][3]))[0])
            y_train.append(tmp)
        x_train = [to_array(batch[i][0]) for i in range(len(batch))]
        return x_train, y_train

    def udpate_eps(self, eps_init, eps_final, t, T):
        return eps_init * (1- t/T) + eps_final*(t/T)

    # max_t : max time alive before reseting the environement
    def train(self, gamma, eps_init, eps_final, T, mini_batch_size, C, max_t):
        print('Training DQN for T={0}, C={1}'.format(T, C))
        # last_q_table = self.get_q_table()
        self.cur_observation = self.env.reset()
        log_freq = 100
        average_loss = 0
        eps = eps_init
        for t in range(T):
            if not t:
                self.fill_D(eps, self.D_size, max_t)
            else:
                self.fill_D(eps, mini_batch_size, max_t)
            mini_batch = self.get_mini_batch(mini_batch_size)

            # y_train will contain the prediction of x_train except for one index so that the loss is as in the paper
            x_train, y_train = self.get_training_set(mini_batch, gamma)
            loss = self.QNN.train_on_batch(x_train, np.array(y_train))[0]
            average_loss += loss

            # Copy QNN weights to TNN's
            if t and not(t % C):
                self.TNN.set_weights(self.QNN.get_weights())
            eps = self.udpate_eps(eps_init, eps_final, t, T)
            # DEBUG
            if t % log_freq == 0:
                print(t)
                # for k,v in self.get_q_table().items():
                #     print(k, v - last_q_table[k])
                # print()

    # Observe the agent on one episode, performs a random action with proportion eps
    def observe(self, rep=1, eps=0.1):
        for t in range(rep):
            observation = self.env.reset()
            while(True):
                self.env.render()
                if random.random() < eps:
                    action = random.randint(0, self.env.action_size-1)
                else:
                    probas = self.QNN.predict(to_array(observation))[0]
                    print(probas)
                    action = np.argmax(probas)
                observation, reward, done, info = self.env.step(action)
                if done:
                    break

    # Outputs a dictionnary of the predictions of every possible position of pacman on the initial map
    def get_q_table(self):
        q_table = dict()
        for i in range(len(self.env.init_map)):
            for j in range(len(self.env.init_map[0])):
                if self.env.init_map[i][j] == 0:
                    q_table[(i, j)] = self.QNN.predict(to_array((self.env.init_map, (i, j), [])))[0]
        return q_table


# To comply with keras format : flatten and transform in np.array
def to_array(observation):
    # high_dim
    if len(observation) == 3:
        ghost_positions = list(itertools.chain.from_iterable(observation[2]))
        return np.array([list(observation[0].flatten()) + list(observation[1]) + ghost_positions])
    else:
        ghost_positions = list(itertools.chain.from_iterable(observation[1]))
        return np.array([list(observation[0]) + ghost_positions])

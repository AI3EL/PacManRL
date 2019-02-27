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

    @staticmethod
    # TODO : set weights to random if not already
    def build_QNN(input_dim):
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dense(4, activation='softmax', name='dense_softmax'))
        model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.RMSprop(),
            metrics=['accuracy']
        )
        return model

    # Fills D with random actions when nothing is known
    # TODO : maybe fill D with the prediction of the initail weights of the QNN
    def init_D(self):
        while len(self.D) < self.D_size:
            observation = self.env.reset()
            while len(self.D) < self.D_size:
                action = random.randint(0, self.env.action_size-1)
                next_observation, reward, info, done = self.env.step(action)
                self.D.append((observation, action, reward, next_observation))
                if done:
                    break

    def get_mini_batch(self, size):
        return [self.D.pop(random.randint(0, len(self.D)-1)) for i in range(size)]

    # TODO: adapt to mini_batch_size != 1 : add more experiences to D
    def train(self, gamma, eps, T, mini_batch_size, C):
        print('Training DQN for T={0}, C={1}'.format(T,C))
        observation = self.env.reset()
        self.init_D()
        for t in range(T):
            if random.random() < eps:
                action = random.randint(0, self.env.action_size - 1)
            else:
                action = np.argmax(self.QNN.predict(to_array(observation), batch_size=1))
            next_observation, reward, done, info = self.env.step(action)
            self.D.append((observation, action, reward, next_observation))  # TODO : know whether episode finished
            mini_batch = self.get_mini_batch(mini_batch_size)
            # y_train will contain the prediction of x_train except for one index so that the loss is as in the paper
            y_train = []
            for i in range(mini_batch_size):
                tmp = self.QNN.predict(to_array(mini_batch[i][0]))[0]  # NOT optimal but simpler with keras
                tmp[mini_batch[i][1]] = mini_batch[i][2] + gamma * max(self.TNN.predict(to_array(mini_batch[i][3]))[0])
                y_train.append(tmp)
            x_train = [to_array(mini_batch[i][0]) for i in range(mini_batch_size)]
            self.QNN.train_on_batch(x_train, np.array(y_train))
            # Copy QNN weights to TNN's
            if t and not(t % C):
                self.TNN.set_weights(self.QNN.get_weights())
            observation = next_observation

    # Observe the agent on one episode, performs a random action with proportion eps
    def observe(self, eps=0.1, max_t=200):
        observation = self.env.reset()
        for t in range(max_t):
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


# To comply with keras format : flatten and transform in np.array
def to_array(observation):
    ghost_positions = list(itertools.chain.from_iterable(observation[2]))
    return np.array([list(observation[0].flatten()) + list(observation[1]) + ghost_positions])
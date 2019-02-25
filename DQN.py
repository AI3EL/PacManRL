import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import keras.backend as K


class DQN:
    def __init__(self, env, buffer_size):
        self.env = env
        self.QNN = self.build_QNN(env.observation_shape)
        self.TNN = self.build_QNN(env.observation_shape)
        self.D = set()
        self.D_size = buffer_size

    @staticmethod
    # TODO : set weights to random if not already
    def build_QNN(input_shape):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape, name='flatten'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(4, activation='softmax', name='dense_softmax'))
        model.compile(
            loss=QNN_loss,
            optimizer=keras.optimizers.RMSprop(),
            metrics=['accuracy']
        )
        return model

    # Fills D with random actions
    def init_D(self):
        while len(self.D) < 100:
            observation = self.env.reset()
            while len(self.D) < 100:
                action = random.randint(0, self.env.action_size-1)
                next_observation, reward, info, done = self.env.step(action)
                self.D.add((observation, action, reward, next_observation))  # TODO : change to list because unhashable type nd array
                if done:
                    break

    def get_mini_batch(self, size):
        return [self.D.pop() for i in range(size)]

    def train(self, gamma, eps, T, mini_batch_size, C):
        observation = self.env.reset()
        self.init_D()
        for t in range(T):
            if random.random() < eps:
                action = random.randint(0, self.env.action_size - 1)
            else:
                action = np.argmax(self.QNN.predict(observation))
            next_observation, reward, done, info = self.env.step(action)
            self.D.add((observation, action, reward, next_observation))  # TODO : know whether episode finished
            mini_batch = self.get_mini_batch(mini_batch_size)
            y_train = [(mini_batch[i][1], mini_batch[i][2] + gamma * max(self.TNN.predict(mini_batch[i][3]))) for i in range(mini_batch_size)]
            x_train = [self.QNN.predict(mini_batch[i][0]) for i in range(mini_batch_size)]
            self.QNN.train_on_batch(y_train, x_train)
            if t and not(t % C):
                self.TNN.set_weights(self.QNN.get_weights())
            observation = next_observation


# y_target should be (a_j, y_j) and y_pred a list containing the Q_theta(obs_j,a) for each a
# TODO : change
def QNN_loss(y_target, y_pred):
    return K.square(y_pred - y_target)

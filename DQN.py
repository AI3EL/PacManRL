import numpy as np
import random
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D
import copy


# Parameters from paper:
# batch_size : 32
# D_size : 10^6
# C : 10^4
# TODO : ideally, every prediction of map has same value for each action at the birth of NN
# TODO : adaptative epsilon
class DQN:
    def __init__(self, env, buffer_size, neurons):
        self.is_conv = False  # To adapt format
        self.env = env
        self.QNN = self.build_QNN(neurons)  # Q network
        print('Network sturcutre :')
        self.QNN.summary()
        self.TNN = self.build_QNN(neurons)  # Target network
        self.D = []  # Pool containing experiences
        self.D_size = buffer_size
        self.eps = None

        # Variables used to recall the environment state when filling D
        self.cur_observation = self.env.reset()
        self.time_alive = 0

        # Log variables
        self.cur_score = 0
        self.average_score = 0
        self.n_episode_since_log = 0
        self.n_train_episodes = 0

    # Only saves NN weights but could also save other things : D, logs, ...
    def save(self, file_name):
        self.QNN.save('models/' + file_name + '.h5')
        self.TNN.save('models/' + file_name + 'T.h5')

    def load(self, file_name):
        self.QNN = load_model('models/' + file_name + '.h5')
        self.TNN = load_model('models/' + file_name + 'T.h5')

    def build_QNN(self, neurons):
        model = Sequential()
        first = True
        shape = self.env.map.shape
        if 'conv' in neurons:
            self.is_conv = True
            for n in neurons['conv']:
                if first:
                    model.add(Conv2D(n, 3, input_shape=(shape[0], shape[1], 1), activation="relu"))
                    first = False
                else:
                    model.add(Conv2D(n, 3, activation="relu"))
            model.add(Flatten())
        else:
            model.add(Flatten(input_shape=shape))
        if 'dense' in neurons:
            for n in neurons['dense']:
                    model.add(Dense(n, activation='relu'))
        model.add(Dense(4, activation='softmax', name='final'))
        model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.RMSprop(),
            metrics=['accuracy']
        )
        return model

    def get_batch(self, size):
        return [self.D.pop(random.randint(0, len(self.D)-1)) for i in range(size)]  # TODO : optimize

    def observations_to_keras(self, observations):
        if self.is_conv:
            return np.array(observations).reshape((len(observations), self.env.map.shape[0], self.env.map.shape[1],1))
        else:
            return np.array(observations)

    def fill_D(self, n, max_t):
        count = 0
        while len(self.D) < self.D_size and count < n:
            # Choosing and performing action
            if random.random() < self.eps:
                action = random.randint(0, self.env.action_size - 1)
            else:
                action = np.argmax(self.QNN.predict(self.observations_to_keras([self.cur_observation]), batch_size=1))
            next_observation, reward, done, info = self.env.step(action)
            self.cur_score += reward
            count += 1

            if done or self.time_alive >= max_t:
                self.D.append((self.cur_observation, action, reward, None))
                self.cur_observation = copy.deepcopy(self.env.reset())
                self.n_episode_since_log += 1
                self.average_score += self.cur_score
                self.time_alive = 0
                self.cur_score = 0
            else:
                self.D.append((self.cur_observation, action, reward, next_observation))
                self.cur_observation = copy.deepcopy(next_observation)

            self.time_alive += 1

    # Transform batch to get training set
    def get_training_set(self, batch, gamma):
        y_train = []
        for i in range(len(batch)):
            # TODO : not optimal but simpler with keras :
            # TODO : just change one element of array tmp so that the error is on only one term of the resulting array
            tmp = self.QNN.predict(self.observations_to_keras([batch[i][0]]))[0]
            if batch[i][3] is None:
                tmp[batch[i][1]] = batch[i][2]
            else:
                tmp[batch[i][1]] = batch[i][2] + gamma * max(self.TNN.predict(self.observations_to_keras([batch[i][3]]))[0])
            y_train.append(tmp)
        x_train = [b[0] for b in batch]
        return self.observations_to_keras(x_train), np.array(y_train)

    # Linearly from eps_init to eps_final for t in [0, T/2], then constant to eps_final
    def udpate_eps(self, eps_init, eps_final, eps_prop, t, T):
        Tf = eps_prop*T
        if t >= Tf:
            self.eps = eps_final
        else:
            self.eps = eps_init * (1- t/Tf) + eps_final*(t/Tf)

    # Train for several epochs
    def train(self, eps_schedule, gamma, T, mini_batch_size, C, max_t, log_freq, save_file=None):
        logs = []
        epochs = len(eps_schedule)
        for i in range(epochs):
            print('Launchin epoch ', i)
            print()
            eps_init, eps_final, eps_prop = eps_schedule[i]
            logs.append(self.train_epoch(gamma, eps_init, eps_final, eps_prop, T, mini_batch_size, C, max_t, log_freq))
            if save_file is not None:
                self.save(save_file)
        return logs

    # Train for one epoch, max_t is the max duration of an episode
    def train_epoch(self, gamma, eps_init, eps_final, eps_prop, T, batch_size, C, max_t, log_freq):
        print('Training DQN for T={}, C={}, batch_size={}'.format(T, C, batch_size))
        # last_q_table = self.get_q_table()
        average_loss = 0
        self.eps = eps_init
        logs = []
        for t in range(T):
            if not t:
                self.fill_D(self.D_size, max_t)
            else:
                self.fill_D(batch_size, max_t)
            batch = self.get_batch(batch_size)

            # y_train will contain the prediction of x_train except for one index so that the loss is as in the paper
            x_train, y_train = self.get_training_set(batch, gamma)
            loss = self.QNN.train_on_batch(x_train, y_train)[0]
            average_loss += loss

            # Copy QNN weights to TNN's
            if t and not(t % C):
                self.TNN.set_weights(self.QNN.get_weights())
            self.udpate_eps(eps_init, eps_final, eps_prop, t, T)

            # Log
            if t % (T/log_freq) == 0:
                logs.append(self.log(t/T))
        return logs

    # Observe the agent on 'rep' episodes, performs a random action with proportion eps
    def observe(self, rep=1, eps=0.1):
        for t in range(rep):
            observation = self.env.reset()
            while True:
                self.env.render()
                if random.random() < eps:
                    action = random.randint(0, self.env.action_size-1)
                else:
                    probas = self.QNN.predict(self.observations_to_keras([observation]))[0]
                    action = np.argmax(probas)
                observation, reward, done, info = self.env.step(action)
                if done:
                    break

    # Outputs a dictionnary of the predictions of every possible position of pacman on the initial map
    def get_q_table(self, map):
        return self.QNN.predict(self.observations_to_keras([map]))[0]

    def log(self, percent):
        print('{}%'.format(int(percent * 100)))
        self.n_train_episodes += self.n_episode_since_log
        if self.n_episode_since_log:
            self.average_score /= self.n_episode_since_log
        print('Training Episodes', self.n_episode_since_log)
        print('Average Score {:0.2f}'.format(self.average_score))
        print('Epsilon {:0.2f}'.format(self.eps))
        self.n_episode_since_log = 0
        print()
        return self.average_score
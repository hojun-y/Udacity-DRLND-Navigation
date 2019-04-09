from collections import deque
import numpy as np
import random


class StateBuilder:
    def __init__(self, history_len=1):
        self._state = deque(maxlen=history_len)
        self._history_len = history_len

    def __call__(self, *args, **kwargs):
        return np.array(self._state)

    def __repr__(self):
        return "State shape=" + str(np.shape(self._state))

    def reset(self, observation):
        for i in range(self._history_len):
            self._state.append(observation)

    def append(self, observation):
        self._state.append(observation)

    def get_state(self):
        return np.array(self._state)


class ReplayMemory:
    def __init__(self, memory_size):
        self._memory = deque(maxlen=memory_size)

    def __repr__(self):
        return "Len: " + str(len(self._memory))

    def append(self, state, action, reward, next_state, done):
        self._memory.append([state, action, reward, next_state, done])

    def get_batch(self, batch_size):
        try:
            batch = random.sample(self._memory, k=batch_size)
        except ValueError:
            raise Warning("Replay memory's length is less than batch size.")

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for data in batch:
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            dones.append(data[4])

        return states, actions, rewards, next_states, dones

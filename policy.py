import random
import numpy as np

def epsilon_greedy(epsilon, actions):
    selected_action = np.zeros_like(actions)
    for i in xrange(actions.shape[0]):
        rn = random.random()
        if rn < epsilon:
            indices = [random.randint(0, actions[i].shape[j] - 1) for j in xrange(len(actions[i].shape))]
            selected_action[(i, ) + tuple(indices)] = 1
        else:
            indices = np.unravel_index(np.argmax(actions[i]), actions[i].shape)
            selected_action[(i, ) + tuple(indices)] = 1
    return selected_action

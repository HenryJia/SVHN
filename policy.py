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

def softmax_policy(actions):
    actions_flat = np.reshape(actions, (actions.shape[0], np.prod(actions.shape[1:])))
    selected_action = np.zeros_like(actions_flat)
    for i in xrange(actions.shape[0]):
        selected_action[i, np.random.choice(selected_action.shape[1], 1, p = actions_flat[i])] = 1
    selected_action = np.reshape(selected_action, actions.shape)
    return selected_action

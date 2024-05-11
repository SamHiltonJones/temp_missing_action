import pickle
import numpy as np
from grid_env import GridWorldEnv

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(file_path, experiences, data_dict):
    with open(file_path, 'wb') as f:
        pickle.dump((experiences, {'pkl': data_dict}), f)

def modify_actions_in_upper_left_room(data, grid_bounds):
    modified_experiences = []
    modified_data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'terminations': []
    }

    x_min, x_max, y_min, y_max = grid_bounds

    for idx, state in enumerate(data['states']):
        if x_min <= state[0] <= x_max and y_min <= state[1] <= y_max:
            action = GridWorldEnv.DOWN 
        else:
            action = data['actions'][idx] 

        modified_experiences.append((
            data['states'][idx],
            action,
            data['rewards'][idx],
            data['next_states'][idx],
            data['terminations'][idx]
        ))

        modified_data['states'].append(data['states'][idx])
        modified_data['actions'].append(action)
        modified_data['rewards'].append(data['rewards'][idx])
        modified_data['next_states'].append(data['next_states'][idx])
        modified_data['terminations'].append(data['terminations'][idx])

    return modified_experiences, modified_data

if __name__ == '__main__':
    data = load_data('core/complete_data_mixed.pkl')

    upper_left_bounds = (1, 5, 1, 5)

    modified_experiences, modified_data = modify_actions_in_upper_left_room(data, upper_left_bounds)

    save_data('core/complete_data_missing.pkl', modified_experiences, modified_data)

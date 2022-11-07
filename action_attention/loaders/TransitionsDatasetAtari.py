import os
import pickle
import numpy as np
from skimage.io import imread
from .Dataset import Dataset


class TransitionsDatasetAtari(Dataset):
    # This data loader is used during training.
    STATE_TEMPLATE = os.path.join("e_{:d}", "s_t_{:d}.npy")
    ACTIONS_TEMPLATE = "actions.pkl"
    POSITIONS_TEMPLATE = "positions.pkl"

    def __init__(self, root_path, factored_actions=True):

        super().__init__()
        self.root_path = root_path
        self.factored_actions = factored_actions
        self.idx2episode = []
        self.num_steps = 0
        self.actions = None
        self.positions = None

        self.load_actions()
        self.load_positions()
        self.build_idx2episode()

    def __getitem__(self, idx):

        ep, step = self.idx2episode[idx]
        obs = self.preprocess_image(self.load_image(ep, step))
        next_obs = self.preprocess_image(self.load_image(ep, step + 1))

        action = self.actions[ep][step]

        return obs, action, next_obs

    def build_idx2episode(self):

        for ep in range(len(self.actions)):
            num_steps = len(self.actions[ep])
            idx_tuple = [(ep, step) for step in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            self.num_steps += num_steps

    def load_actions(self):

        load_path = os.path.join(self.root_path, self.ACTIONS_TEMPLATE)
        if not os.path.isfile(load_path):
            raise ValueError("Actions not found.")
        with open(load_path, "rb") as f:
            self.actions = pickle.load(f)

    def load_positions(self):

        load_path = os.path.join(self.root_path, self.POSITIONS_TEMPLATE)
        if not os.path.isfile(load_path):
            raise ValueError("Positions not found.")
        with open(load_path, "rb") as f:
            self.positions = pickle.load(f)

    def load_image(self, ep, step):

        load_path = os.path.join(self.root_path, self.STATE_TEMPLATE.format(ep, step))
        return np.load(load_path)

    def obj_action_to_pos_action(self, ep, step):

        action = self.actions[ep][step]
        obj_idx = action // 4
        dir_idx = action % 4

        position = self.positions[ep][step][obj_idx]
        new_action = np.zeros(5 + 5 + 4, dtype=np.float32)
        new_action[position[0]] = 1
        new_action[position[1] + 5] = 1
        new_action[dir_idx + 10] = 1
        return new_action

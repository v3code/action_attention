import pickle
import os
import numpy as np
from skimage.io import imread
from .Dataset import Dataset


class PathDatasetAtari(Dataset):
    # This data loader is used during evaluation.
    STATE_TEMPLATE = os.path.join("e_{:d}", "s_t_{:d}.npy")
    ACTIONS_TEMPLATE = "actions.pkl"
    STATE_IDS_TEMPLATE = "state_ids.pkl"

    def __init__(self, root_path, path_length, factored_actions=True):

        super().__init__()
        self.root_path = root_path
        self.path_length = path_length
        self.factored_actions = factored_actions
        self.actions = None
        self.state_ids = None

        self.load_actions()
        self.load_state_ids()
        self.num_steps = len(self.actions)

    def __getitem__(self, ep):

        obs = []
        actions = []
        state_ids = []

        for step in range(self.path_length):

            obs.append(self.preprocess_image(self.load_image(ep, step)))
            action = self.actions[ep][step]
            actions.append(action)
            state_id = self.state_ids[ep][step]
            state_ids.append(state_id)

        obs.append(self.preprocess_image(self.load_image(ep, self.path_length)))
        state_ids.append(self.state_ids[ep][self.path_length])

        return obs, actions, state_ids

    def load_actions(self):

        load_path = os.path.join(self.root_path, self.ACTIONS_TEMPLATE)
        if not os.path.isfile(load_path):
            raise ValueError("Actions not found.")
        with open(load_path, "rb") as f:
            self.actions = pickle.load(f)

    def load_state_ids(self):

        load_path = os.path.join(self.root_path, self.STATE_IDS_TEMPLATE)
        if not os.path.isfile(load_path):
            raise ValueError("Positions not found.")
        with open(load_path, "rb") as f:
            self.state_ids = pickle.load(f)

    def load_image(self, ep, step):

        load_path = os.path.join(self.root_path, self.STATE_TEMPLATE.format(ep, step))
        return np.load(load_path)

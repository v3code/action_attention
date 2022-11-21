import copy as cp
import os
import pickle
import numpy as np
import gym
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.io import imsave
from ...constants import Constants
from ...stack import StackElement
from ... import utils
from skimage.transform import resize


class InitEnvAndSeed(StackElement):
    # Initialize the environment and set the random seed.
    def __init__(self, env_id, seed):

        super().__init__()
        self.env_id = env_id
        self.seed = seed
        self.OUTPUT_KEYS = {Constants.ENV}

    def run(self, bundle: dict, viz=False) -> dict:

        env = gym.make(self.env_id)

        np.random.seed(self.seed)
        env.action_space.seed(self.seed)
        env.seed(self.seed)

        return {
            Constants.ENV: env
        }
def crop_normalize(img, crop_ratio):
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Image.ANTIALIAS)
    return np.array(img) / 255

prepro = lambda img: resize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def preprocess_state(state):

    return torch.tensor(prepro(state))
def select_action(state, model, hx, eps):
    # select an action using either an epsilon greedy or softmax policy
    value, logit, hx = model((state.view(1, 1, 80, 80), hx))
    logp = F.log_softmax(logit, dim=-1)

    if eps is not None:
        # use epsilon greedy
        if np.random.uniform(0, 1) < eps:
            # random action
            return np.random.randint(logp.size(1))
        else:
            return torch.argmax(logp, dim=1).numpy()[0]
    else:
        # sample from softmax
        action = torch.exp(logp).multinomial(num_samples=1).data[0]
        return action.numpy()[0]

def reset_rnn_state():
    # reset the hidden state of an rnn
    return torch.zeros(1, 256)

def construct_start_states_set(dup_paths):
    # go through a dataset and find all start states
    # we use this to reduce the overlap between train/valid/test sets
    blacklist_state_ids = set()

    for path in dup_paths:

        f = h5py.File(path, "r")

        # iterate over all episodes, stored in a dict
        for ep in f.values():
            # cheap way of making an immutable array
            blacklist_state_ids.add(ep['state_ids'][0].tobytes())

        f.close()

    return blacklist_state_ids

class CollectA3CAtariAndSave(StackElement):
    # Collect data using a random policy. Save states as PNG images, and actions and positions as pickles.
    STATE_TEMPLATE = os.path.join("e_{:d}", "s_t_{:d}.npy")
    ACTIONS_TEMPLATE = "actions.pkl"
    STATE_IDS_TEMPLATE = "state_ids.pkl"

    def __init__(self, save_path, num_episodes, num_steps, min_burnin, max_burnin, eps=0.5, crop=None, factored_actions=True, ):

        super().__init__()
        self.save_path = save_path
        self.num_episodes = num_episodes
        self.factored_actions = factored_actions
        self.num_steps = num_steps
        self.min_burnin = min_burnin
        self.max_burnin = max_burnin
        self.crop = crop
        self.eps = eps
        self.INPUT_KEYS = {Constants.ENV, Constants.A3C}

    def run(self, bundle: dict, viz=False) -> dict:

        if os.path.isdir(self.save_path):
            raise ValueError("Save path already occupied.")

        env = bundle[Constants.ENV]
        a3c = bundle[Constants.A3C]

        actions = [[] for _ in range(self.num_episodes)]
        state_ids = [[] for _ in range(self.num_episodes)]



        for ep_idx in range(self.num_episodes):
            burnin_steps = np.random.randint(self.min_burnin, self.max_burnin)
            hx = reset_rnn_state()

            prev_obs = env.reset()
            step_idx = 0
            for _ in range(burnin_steps):
                action = select_action(preprocess_state(prev_obs), a3c, hx, self.eps)
                prev_obs, _, _, _ = env.step(action)



            if self.crop:
                prev_obs = crop_normalize(prev_obs, self.crop)

            obs, _, _, _ = env.step(0)

            if self.crop:
                obs = crop_normalize(obs, self.crop)

            self.save_obs(ep_idx, step_idx, obs, prev_obs)
            prev_obs = obs

            state_ids[ep_idx].append(cp.deepcopy(np.array(env.unwrapped._get_ram(), dtype=np.int32)))

            while True:


                # select random action
                action = env.action_space.sample()
                obs, _, done, _ = env.step(action)


                if self.crop:
                    obs = crop_normalize(obs, self.crop)
                actions[ep_idx].append(action)
                step_idx += 1
                self.save_obs(
                    ep_idx, step_idx,
                    obs, prev_obs
                )
                state_ids[ep_idx].append(cp.deepcopy(np.array(env.unwrapped._get_ram(), dtype=np.int32)))
                prev_obs = obs

                if step_idx >= self.num_steps:
                    break

                if done:
                    break

            if ep_idx > 0 and ep_idx % 10 == 0:
                self.logger.info("episode {:d}".format(ep_idx))

        self.save_actions(actions)
        self.save_state_ids(state_ids)

        return {}

    def save_obs(self, ep, step, obs, prev_obs):

        save_path = os.path.join(self.save_path, self.STATE_TEMPLATE.format(ep, step))
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        result = np.concatenate((obs, prev_obs), axis=2)
        np.save(save_path, result)


    def save_actions(self, actions):

        save_path = os.path.join(self.save_path, self.ACTIONS_TEMPLATE)
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(actions, f)

    def save_state_ids(self, state_ids):

        save_path = os.path.join(self.save_path, self.STATE_IDS_TEMPLATE)
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(state_ids, f)


class CollectRandomAtariAndSave(StackElement):
    # Collect data using a random policy. Save states as PNG images, and actions and positions as pickles.
    STATE_TEMPLATE = os.path.join("e_{:d}", "s_t_{:d}.npy")
    ACTIONS_TEMPLATE = "actions.pkl"
    STATE_IDS_TEMPLATE = "state_ids.pkl"

    def __init__(self, save_path, num_episodes, num_steps, crop=None, warmstart=None, factored_actions=True, ):

        super().__init__()
        self.save_path = save_path
        self.num_episodes = num_episodes
        self.factored_actions = factored_actions
        self.num_steps = num_steps
        self.crop = crop
        self.warmstart = warmstart
        self.INPUT_KEYS = {Constants.ENV}

    def run(self, bundle: dict, viz=False) -> dict:

        if os.path.isdir(self.save_path):
            raise ValueError("Save path already occupied.")

        env = bundle[Constants.ENV]
        actions = [[] for _ in range(self.num_episodes)]
        state_ids = [[] for _ in range(self.num_episodes)]

        for ep_idx in range(self.num_episodes):

            prev_obs = env.reset()
            step_idx = 0
            if self.warmstart:
                for _ in range(self.warmstart):
                    action = env.action_space.sample()
                    prev_obs, _, _, _ = env.step(action)



            if self.crop:
                prev_obs = crop_normalize(prev_obs, self.crop)

            obs, _, _, _ = env.step(0)

            if self.crop:
                obs = crop_normalize(obs, self.crop)

            self.save_obs(ep_idx, step_idx, obs, prev_obs)
            prev_obs = obs

            state_ids[ep_idx].append(cp.deepcopy(np.array(env.unwrapped._get_ram(), dtype=np.int32)))

            while True:


                # select random action
                action = env.action_space.sample()
                obs, _, done, _ = env.step(action)


                if self.crop:
                    obs = crop_normalize(obs, self.crop)
                actions[ep_idx].append(action)
                step_idx += 1
                self.save_obs(
                    ep_idx, step_idx,
                    obs, prev_obs
                )
                state_ids[ep_idx].append(cp.deepcopy(np.array(env.unwrapped._get_ram(), dtype=np.int32)))
                prev_obs = obs

                if step_idx >= self.num_steps:
                    break

                if done:
                    break

            if ep_idx > 0 and ep_idx % 10 == 0:
                self.logger.info("episode {:d}".format(ep_idx))

        self.save_actions(actions)
        self.save_state_ids(state_ids)

        return {}

    def save_obs(self, ep, step, obs, prev_obs):

        save_path = os.path.join(self.save_path, self.STATE_TEMPLATE.format(ep, step))
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        result = np.concatenate((obs, prev_obs), axis=2)
        np.save(save_path, result)


    def save_actions(self, actions):

        save_path = os.path.join(self.save_path, self.ACTIONS_TEMPLATE)
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(actions, f)

    def save_state_ids(self, state_ids):

        save_path = os.path.join(self.save_path, self.STATE_IDS_TEMPLATE)
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(state_ids, f)

class CollectRandomAndSave(StackElement):
    # Collect data using a random policy. Save states as PNG images, and actions and positions as pickles.
    STATE_TEMPLATE = os.path.join("e_{:d}", "s_t_{:d}.png")
    ACTIONS_TEMPLATE = "actions.pkl"
    POSITIONS_TEMPLATE = "positions.pkl"

    def __init__(self, save_path, num_episodes, factored_actions=True):

        super().__init__()
        self.save_path = save_path
        self.num_episodes = num_episodes
        self.factored_actions = factored_actions
        self.INPUT_KEYS = {Constants.ENV}

    def run(self, bundle: dict, viz=False) -> dict:

        if os.path.isdir(self.save_path):
            raise ValueError("Save path already occupied.")

        env = bundle[Constants.ENV]
        actions = [[] for _ in range(self.num_episodes)]
        positions = [[] for _ in range(self.num_episodes)]

        for ep_idx in range(self.num_episodes):

            obs = env.reset()
            step_idx = 0
            self.save_image(
                ep_idx, step_idx,
                utils.float_0_1_image_to_uint8(obs[1])
            )
            positions[ep_idx].append(cp.deepcopy(obs[0]))

            while True:

                # select random action
                action = env.action_space.sample()
                obs, _, done, _ = env.step(action)

                actions[ep_idx].append(action)
                step_idx += 1
                self.save_image(
                    ep_idx, step_idx,
                    utils.float_0_1_image_to_uint8(obs[1])
                )
                positions[ep_idx].append(cp.deepcopy(obs[0]))

                if done:
                    break

            if ep_idx > 0 and ep_idx % 10 == 0:
                self.logger.info("episode {:d}".format(ep_idx))

        self.save_actions(actions)
        self.save_positions(positions)

        return {}

    def save_image(self, ep, step, img):

        save_path = os.path.join(self.save_path, self.STATE_TEMPLATE.format(ep, step))
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        imsave(save_path, img)

    def save_actions(self, actions):

        save_path = os.path.join(self.save_path, self.ACTIONS_TEMPLATE)
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(actions, f)

    def save_positions(self, positions):

        save_path = os.path.join(self.save_path, self.POSITIONS_TEMPLATE)
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(positions, f)

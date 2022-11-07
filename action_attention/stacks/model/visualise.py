from PIL import Image
import numpy as np
import io
import torch
import matplotlib.pyplot as plt
from action_attention import utils
from action_attention.constants import Constants
from action_attention.stack import StackElement


class Visualise(StackElement):

    def __init__(self, device, save_path, viz_name, seed):

        super(Visualise, self).__init__()

        self.device = device
        self.save_path = save_path
        self.viz_name = viz_name
        self.seed = seed
        np.random.seed(seed)

        self.INPUT_KEYS = {Constants.MODEL, Constants.ENV}

    def _img_to_torch(self, img):
        return torch \
            .from_numpy(np.array([img])) \
            .to(self.device).permute((0, 3, 1, 2)) \
            .float()

    def _action_to_torch(self, action, pos):
        obj_idx = action // 4
        dir_idx = action % 4

        position = pos[obj_idx]
        new_action = np.zeros(5 + 5 + 4, dtype=np.float32)
        new_action[position[0]] = 1
        new_action[position[1] + 5] = 1
        new_action[dir_idx + 10] = 1
        return torch.from_numpy(np.array([new_action])).to(self.device)

    def run(self, bundle: dict, viz=False) -> dict:

        images = []

        env = bundle[Constants.ENV]
        model = bundle[Constants.MODEL]
        model.eval()

        has_attention = hasattr( model, 'use_attention') and model.use_attention

        with torch.no_grad():

            obs = env.reset()
            done = False
            step = 1
            obs_image = self._img_to_torch(obs[1])
            state = model.forward(obs_image)
            prev_pred_state = state
            while not done:
                action = env.action_space.sample()
                next_obs, _, done, _ = env.step(action)
                next_obs_image = self._img_to_torch(next_obs[1])
                next_state = model.forward(next_obs_image)
                action = self._action_to_torch(action, obs[0])
                if has_attention:
                    weights = model.forward_weights(state, action)
                # action = self._action_to_torch(action, obs[0]) if has_attention else torch.Tensor([action]).long()
                pred_state = model.forward_transition(prev_pred_state, action)

                fig = plt.figure(figsize=(10, 5))
                plt.suptitle(f"{self.viz_name}, step = {step}", fontsize=16)

                plt.subplot(2, 3, 1)
                plt.title("current state")
                if len(obs_image.shape) == 4:
                    plt.imshow(obs_image.cpu().numpy().sum(0).transpose((1, 2, 0)))
                else:
                    plt.imshow(obs_image.cpu().numpy().transpose((1, 2, 0)))
                plt.subplot(2, 3, 2)
                plt.title("next state")
                if len(next_obs_image.shape) == 4:
                    plt.imshow(next_obs_image.cpu().numpy().sum(0).transpose((1, 2, 0)))
                else:
                    plt.imshow(next_obs_image.cpu().numpy().transpose((1, 2, 0)))

                if has_attention:
                    plt.subplot(2, 3, 3)
                    plt.title("predicted attention weights")
                    plt.bar(list(range(1, len(weights[0]) + 1)), weights[0].cpu().numpy())

                plt.subplot(2, 3, 4)
                plt.title("encoding of current state")
                for j in range(5):
                    plt.scatter(
                        [state[0, j, 0].cpu().numpy()],
                        [state[0, j, 1].cpu().numpy()]
                    )
                plt.subplot(2, 3, 5)
                plt.title("encoding of next state")
                for j in range(5):
                    plt.scatter(
                        [next_state[0, j, 0].cpu().numpy()],
                        [next_state[0, j, 1].cpu().numpy()]
                    )
                plt.subplot(2, 3, 6)
                plt.title("predicted next state")
                for j in range(5):
                    plt.scatter(
                        [pred_state[0, j, 0].cpu().numpy()],
                        [pred_state[0, j, 1].cpu().numpy()]
                    )
                plt.tight_layout()
                fig.canvas.draw()
                images.append(self._plt_to_pil(fig))
                plt.close()
                step += 1
                obs = next_obs
                prev_pred_state = pred_state
                obs_image = next_obs_image
                state = next_state

            self._make_gif(images)
            return {}

    def _plt_to_pil(self, fig):
        return Image.frombytes('RGB',
                               fig.canvas.get_width_height(),
                               fig.canvas.tostring_rgb())

    def _make_gif(self, frames):
        frame_one = frames[0]
        frame_one.save(self.save_path, format="GIF", append_images=frames,
                       save_all=True, duration=len(frames) * 30, loop=0)

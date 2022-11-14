import numpy as np
from torch.utils import data as torch_data


class Dataset(torch_data.Dataset):
    # "Abstract" dataset class used in a pytorch data loader.
    def __init__(self):
        self.num_steps = None
        self.is_atari = False

    def __len__(self):
        return self.num_steps

    def preprocess_image(self, image: np.ndarray):

        image = np.array(image, dtype=np.float32)
        if not self.is_atari:
            image = image / 255.


        if len(image.shape) == 3:
            # a single RGB image
            return image.transpose((2, 0, 1))
        elif len(image.shape) == 4:
            # an RGB image for each object
            return image.transpose((0, 3, 1, 2))
        else:
            raise NotImplementedError()

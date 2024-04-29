import cv2
import numpy as np

from glob import glob
import torch
from torch.utils.data import Dataset

from datasets.utils import clahe_equalized
from datasets.transform import pipeline_tranforms


class CHASEDBDataset(Dataset):

    def __init__(self, CFG, image_path, mask_path):
        self.images_path = sorted(glob(image_path))
        self.masks_path = sorted(glob(mask_path))

        self.transforms = pipeline_tranforms()
        self.size = CFG.size
        self.n_samples = len(self.images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = clahe_equalized(image)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        image = image / 255.0  # (512, 512, 3) Normalizing to range (0,1)
        image = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        try:
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(str(e))
            print(self.masks_path[index])
        
        mask = mask / 255.0  # (512, 512)
        mask = np.expand_dims(mask, axis=0)  # (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

            # common transform
        if self.transforms is not None:
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transforms(image) # type: ignore
            torch.manual_seed(seed)
            mask = self.transforms(mask) # type: ignore

        return image, mask

    def __len__(self):
        return self.n_samples
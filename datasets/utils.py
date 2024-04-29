import numpy as np
import cv2
import os
from glob import glob
from loguru import logger

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


# CLAHE
def clahe_equalized(image):  
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr 

def fives_loader(Dataset, CFG):

    # Split dataset into train and validation
    validation_split = .2
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(Dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(CFG.random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(Dataset, batch_size=CFG.batch_size, pin_memory=True,
                              sampler=train_sampler, drop_last=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(Dataset, batch_size=CFG.batch_size, drop_last=True,
                            sampler=valid_sampler, pin_memory=True, num_workers=CFG.num_workers)

    logger.info(
        'The total number of images for train and validation is %d' % len(Dataset))

    return train_loader, val_loader

def fives_test_loader(Dataset):

    loader = DataLoader(dataset=Dataset, batch_size=1,
                        shuffle=False, pin_memory=True, num_workers=8)

    return loader

def load_subgroup_images(disease, root):
    def load_paths(subdir):
        original = sorted(glob(os.path.join(root, subdir, 'Original/*')))
        ground_truth = sorted(glob(os.path.join(root, subdir, 'Ground truth/*')))

        # Exclude the database file from training images only
        if 'train' in subdir:
            original = original[:-1]

        return original, ground_truth

    train_x, train_y = load_paths('train')
    valid_x, valid_y = load_paths('test')

    # Split into training and validation sets based on the presence of 'disease' in the filename
    def split_data(items):
        return ([item for item in items if disease in os.path.basename(item)],
                [item for item in items if disease not in os.path.basename(item)])

    valid_x, train_x = split_data(train_x + valid_x)
    valid_y, train_y = split_data(train_y + valid_y)

    return train_x, train_y, valid_x, valid_y


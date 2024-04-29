import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class DRIVEDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        
        self.images_path = sorted(glob(data_path + "/images/*"))
        self.masks_path = sorted(glob(data_path + "/labels/*"))
        self.n_samples = len(self.images_path)
        for i in self.masks_path:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, index):

        data = Image.open(self.images_path[index]).convert('RGB').resize((512,512), resample=Image.Resampling.NEAREST)
        label = Image.open(self.masks_path[index]).convert('L').resize((512,512), resample=Image.Resampling.NEAREST)

        data = np.array(data)
        label = np.array(label)

        if data.shape[-1]==3:
            data = torch.from_numpy(np.array(data).transpose(2, 0, 1)).float() / 255
            label = torch.from_numpy(np.array(label)).float().unsqueeze(0) / 255
        else:
            data = torch.from_numpy(data).unsqueeze(0).float() / 255
            label = torch.from_numpy(label).float().unsqueeze(0) / 255

        return data, label
    
    def __len__(self):
        return self.n_samples
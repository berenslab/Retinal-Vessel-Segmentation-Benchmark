import argparse
from glob import glob 
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
import torch
from torch.utils.data import DataLoader

import networks as models
from datasets.fives import FIVES
from datasets.chase import CHASEDBDataset
from datasets.drive import DRIVEDataset
from datasets.utils import fives_loader
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch


def main(CFG):
    seed_torch()
    logger.info(f'RUNNING with the following configurations!!! \n \n {CFG} \n\n')


    if CFG['dataset']['type'] == 'FIVES':
        images = sorted(glob(CFG['dataset']['path'] + f"/train/Original/*"))[:-1]
        masks  = sorted(glob(CFG['dataset']['path'] +f"/train/Ground truth/*"))

        dataset = FIVES(CFG=CFG, images_path=images, mask_paths=masks, mode='train')
        train_loader, val_loader = fives_loader(Dataset=dataset, CFG=CFG)
        
    elif CFG['dataset']['type'] == 'CHASEDB':

        train_dataset = CHASEDBDataset(CFG['dataset']['train_path'] + "/image/*", CFG['dataset']['train_path'] + "/label/*" )
        val_dataset = CHASEDBDataset(CFG['dataset']['valid_path'] + "/images/*", CFG['dataset']['valid_path']  + "/labels/*")

        ## Loader
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, pin_memory=True, drop_last=True, num_workers=CFG.num_workers)
        val_loader  = DataLoader(val_dataset, batch_size=CFG.batch_size, pin_memory=True, drop_last=True, num_workers=CFG.num_workers)

    elif CFG['dataset']['type'] == 'DRIVE':

        train_path = '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/data_aug/Aug_data/DRIVE/train/'
        valid_path = '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/data_aug/Aug_data/DRIVE/validate/'

        train_dataset = DRIVEDataset(train_path)
        val_dataset = DRIVEDataset(valid_path)

        # Loader
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size,
                                pin_memory=False, drop_last=True, num_workers=CFG.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size,
                                pin_memory=False, drop_last=True, num_workers=CFG.num_workers)
        
    else:
        raise NotImplementedError("Dataset type should be either DRIVE | FIVES | CHASEDB ")
        
            

    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader,
        device = device
    )

    trainer.train()
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", help="Configuration file to load", )
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))

    main(CFG)

# python -u src/train.py --config configs/manet.yaml 
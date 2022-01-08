import torch
import cv2
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from augmentation import get_transform_pipeline

class SaltDataset(Dataset):
    def __init__(self, data_dir, transforms=None,):
        self.data_dir=data_dir
        self.images_path = os.path.join(self.data_dir, 'images')
        self.masks_path = os.path.join(self.data_dir, 'masks')
        self.img_dir=sorted(os.listdir(self.images_path))
        self.transform = transforms
        return 

    def __getitem__(self, index: int):
        img_name=self.img_dir[index]
        img_path = os.path.join(self.images_path, img_name)
        mask_path = os.path.join(self.masks_path, img_name)

        image=cv2.imread(img_path)
        mask=cv2.cvtColor(cv2.imread(mask_path),cv2.COLOR_BGR2GRAY)
        if(self.transform is not None):
            aug_dict=self.transform(image=image,mask=mask)
            image=aug_dict['image']
            mask=aug_dict['mask']
        mask=mask.unsqueeze(dim=0)
        return (image, mask)

    def __len__(self) -> int:
        return self.data.shape[0]

if __name__ == '__main__':
    dataset=SaltDataset('Dataset/train',get_transform_pipeline(101,101,False))
    img,mask=dataset[0]
    print(img.shape,mask.shape)
    pass
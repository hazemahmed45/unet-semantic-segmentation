from albumentations import (
    Compose, ShiftScaleRotate, RandomBrightness, HueSaturationValue, RandomContrast, HorizontalFlip,VerticalFlip,
    Rotate, Resize, Lambda, CLAHE, ColorJitter, RandomBrightnessContrast, GaussianBlur, Blur, MedianBlur,
    GridDistortion, Downscale, ChannelShuffle, Normalize, OneOf, IAAAdditiveGaussianNoise, GaussNoise,
    RandomScale
)
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms
import torch
import numpy as np
from numpy import dtype, random
def get_low_aug_transform_pipeline(width, height, is_train=True,prob_range=[0.4,0.6]):
    if(is_train):
        return Compose([
            Rotate(limit=15, p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.78),
            HorizontalFlip(p=random.uniform(low=prob_range[0],high=prob_range[1])),#),
            Downscale(),
            Resize(height=height, width=width),
            Normalize(),
            ToTensorV2()
        ])
    else:
        return Compose([
            Resize(height=height, width=width),
            Normalize(),
            ToTensorV2()

        ])
def get_torch_transform_pipeline(width,height,is_train=True):
    if(is_train):
        return transforms.Compose([
            transforms.Resize((width,height)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        
        return transforms.Compose([
            transforms.Resize((width,height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]) 
def get_transform_pipeline(width, height, is_train=True,prob_range=[0.4,0.6]):
    if(is_train):
        return Compose([
            OneOf([
                ShiftScaleRotate(rotate_limit=15, p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.87),
                Rotate(limit=15, p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.78),
                RandomScale(p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.78)
            ], p=random.uniform(low=prob_range[0],high=prob_range[1])),#0.85),
            HorizontalFlip(p=random.uniform(low=prob_range[0],high=prob_range[1])),#),
            VerticalFlip(p=random.uniform(low=prob_range[0],high=prob_range[1])),
            GaussNoise(p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.67),
            OneOf([
                Blur(p=random.uniform(low=prob_range[0],high=prob_range[1])),#),
                GaussianBlur(p=random.uniform(low=prob_range[0],high=prob_range[1])),#),
                MedianBlur(p=random.uniform(low=prob_range[0],high=prob_range[1])),#)
            ], p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.75),
            OneOf([
                OneOf([
                    RandomBrightness(p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.78),
                    RandomContrast(p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.78)
                ], p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.75),
                OneOf([
                    CLAHE(p=random.uniform(low=prob_range[0],high=prob_range[1])),#),
                    ColorJitter(p=random.uniform(low=prob_range[0],high=prob_range[1])),#),
                    HueSaturationValue(p=random.uniform(low=prob_range[0],high=prob_range[1])),#),
                    ChannelShuffle(p=random.uniform(low=prob_range[0],high=prob_range[1])),#),
                ], p=random.uniform(low=prob_range[0],high=prob_range[1])),#p=0.75),
            ],p=random.uniform(low=prob_range[0],high=prob_range[1])),#0.5),
            #GridDistortion
            #CutMix
            Downscale(scale_min=0.25, scale_max=0.25, p=random.uniform(low=prob_range[0],high=prob_range[1])),
            
                
            Resize(height=height, width=width),
            Normalize(mean=0.5,std=0.5),
            ToTensorV2()


        ])
    else:
        return Compose([
            Resize(height=height, width=width),
            Normalize(),
            ToTensorV2()

        ])


if ( __name__ == '__main__'):
    image=np.random.randint(0,255,(101,101,1),dtype=np.uint8)
    mask=np.random.randint(0,1,(101,101,1),dtype=np.uint8)
    out=get_transform_pipeline(101,101)(image=image,mask=mask)
    print(mask.max(),mask.min())
    print(out['image'],out['mask'],out['mask'].max(),out['mask'].min())
    pass
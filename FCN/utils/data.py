from pathlib import Path
import numbers

import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms.functional as Ftrans


class VOC2012(data.Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        self.root = Path(root)
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        self.images = list(sorted(self.root.joinpath('JPEGImages').glob('*.jpg')))
        self.labels = list(sorted(self.root.joinpath('SegmentationClass').glob('*.png')))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        with open(str(self.images[index]), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(str(self.labels[index]), 'rb') as f:
            label = Image.open(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return image, label


class CenterCropWithIgnore:
    def __init__(self, size, ignore):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
        self.ignore = int(ignore)

    def __call__(self, image):
        Fimage = Ftrans.center_crop(image, self.size)
        
        label = torch.from_numpy(np.array(Fimage)).long().unsqueeze(0)

        W, H = Fimage.size
        Wimg,Himg = image.size
        ys = (H - Himg) // 2
        xs = (W - Wimg) // 2
        
        label[0, 0:ys, 0:] = self.ignore
        label[0, (H-ys):H, 0:] = self.ignore
        label[0, ys:(H-ys), 0:xs] = self.ignore
        label[0, ys:(H-ys), (W-xs):W] = self.ignore
        
        return label

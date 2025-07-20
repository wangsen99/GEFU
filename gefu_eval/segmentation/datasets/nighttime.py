# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:51:21 2019 by Attila Lengyel - attila@lengyel.nl
"""

import os
import numpy as np

from PIL import Image

from torchvision.datasets import Cityscapes
class NighttimeDataset(Cityscapes):
    
    voidClass = 19
    
    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass
    
    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')
    
    def __init__(self, root, transforms=None, add=''):
        self.transforms = transforms
        
        self.root = root
        
        if add != '':
            add = '_' + add
        self.imgs_root = os.path.join(root,'images/gefu_seg'+add)
        self.masks_root = os.path.join(root,'labels/test')

        self.imgs = list(sorted(os.listdir(self.imgs_root)))
        # self.masks = [mask for mask in list(sorted(os.listdir(self.masks_root))) if 'labelIds' in mask]
        self.masks = [mask for mask in list(sorted(os.listdir(self.masks_root)))]

        # print(len(self.imgs), len(self.masks))
        
        assert len(self.imgs) == len(self.masks), 'Number of images and masks must be equal'
        assert len(self.imgs) != 0, 'No images found'
    
    def __getitem__(self, idx):
        # Define image and mask path
        img_path = os.path.join(self.imgs_root, self.imgs[idx])
        mask_path = os.path.join(self.masks_root, self.masks[idx])
        # print(img_path, mask_path)
        image = Image.open(img_path).convert('RGB')
        target = Image.open(mask_path)
        
        if self.transforms is not None:
            image, target, ms_image = self.transforms(image, target)
            
        # Convert ids to train_ids
        # print('target', target)
        # print(np.unique(target))
        # target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW
        voidClass = 19
        target[np.where(target==255)] = voidClass
        return image, target, img_path, ms_image
        
    def __len__(self):
        return len(self.imgs)

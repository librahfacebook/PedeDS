# -*- coding:utf-8 -*-
# @Time: 2020/5/3 21:58
# @Author: libra
# @Site: Define a custom dataset for PennFudan
# @File: dataloader.py
# @Software: PyCharm

import os
import numpy as np
import torch
from PIL import Image

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs_root = os.path.join(root, "PNGImages")
        self.masks_root  = os.path.join(root, "PedMasks")
        self.imgs = list(sorted(os.listdir(self.imgs_root)))
        self.masks = list(sorted(os.listdir(self.masks_root)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_root,self.imgs[idx])
        mask_path = os.path.join(self.masks_root,self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        # the instances are encodes as different colors, including background 0
        object_ids = np.unique(mask)
        object_ids = object_ids[1:]  # remove the background
        # split the color-encodes mask into a set of binary masks
        masks = mask == object_ids[:,None,None]

        # get bounding box coordinates for each mask
        num_objects = len(object_ids)
        boxes = []
        for i in range(num_objects):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin,ymin,xmax,ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objects,),dtype=torch.int64)
        masks = torch.as_tensor(masks,dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        iscrowd = torch.zeros((num_objects,),dtype=torch.int64)

        target = {"boxes":boxes,"labels":labels,"masks":masks,
                  "image_id":image_id,"area":area,"iscrowd":iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img,target)

        return img,target

    def __len__(self):
        return len(self.imgs)





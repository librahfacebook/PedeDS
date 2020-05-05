# -*- coding:utf-8 -*-
# @Time: 2020/5/3 23:44
# @Author: libra
# @Site: build net
# @File: maskNet.py
# @Software: PyCharm

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get numbers of input features
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features,num_classes)

    # get numbers of input features for mask classifier
    input_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(input_features_mask,hidden_layer,num_classes)

    return model

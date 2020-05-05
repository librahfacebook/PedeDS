## Pedestrian detection and Segmentation 

```
---
title: PDS: 行人检测及分割
author: librah
date: 2020/5/5
---
```

#### Introduction

Object Detection is a common Computer Vision problem which deals with identifying and locating object of certain classes in the image. Interpreting the object localization in various ways, including creating a bounding box around the object or marking every pixel in the image which contains the object (called segmentation).

<img src="https://miro.medium.com/max/1096/1*NXWE7BHug0i-FQlHo5xa7w.png" alt="img" style="zoom:50%;" />

Image segmentation creates a pixel-wise mask for each object in the image. This technique gives us a far more granular understanding of the object(s) in the image.

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/image-segmentation.png" alt="img" style="zoom:50%;" />

#### Models

The model is Mask R-CNN, which is based on top of Faster R-CNN. Mask R-CNN adds an extra branch into Faster R-CNN, which also predicts segmentation masks for each instance.

<img src="https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image04.png" alt="intermediate/../../_static/img/tv_tutorial/tv_image04.png" style="zoom:50%;" />

#### Structure

networks: Mask R-CNN, based on torchvision.models.detection. We use the FastRCNNPredictor to detection boxes and scores of objects, and use the MaskRCNNPredictor to object segmentation.

utils: some official application tools.

dataloader.py: define a custom dataset for Pedestrian.

experiments.py:  train model, predict and show the results.

#### Results

![image](https://user-images.githubusercontent.com/34414402/81048152-59500b00-8eee-11ea-9fe1-093417819fe7.png)
![image](https://user-images.githubusercontent.com/34414402/81048168-5fde8280-8eee-11ea-8b72-9639e780e081.png)

#### Reference

[1] https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

[2] https://matplotlib.org/3.1.1/contents.html
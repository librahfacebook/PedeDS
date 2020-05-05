# -*- coding:utf-8 -*-
# @Time: 2020/5/4 1:00
# @Author: libra
# @Site: 
# @File: experiment.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from dataloader import PennFudanDataset
from networks.maskNet import get_model_instance_segmentation
from utils.engine import train_one_epoch,evaluate
from utils import tools
from utils.transforms import get_transform
from torch.utils.data import Subset,DataLoader
from PIL import Image

CLASSES = ['background','person']

def train(data_root):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # background and person
    num_classes = 2
    dataset = PennFudanDataset(data_root,get_transform(train=True))
    dataset_test = PennFudanDataset(data_root,get_transform(train=False))

    # split the dataset
    indices = torch.randperm(len(dataset)).tolist()
    dataset = Subset(dataset,indices[:-50])
    dataset_test = Subset(dataset_test,indices[-50:])

    # define data loaders
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                             collate_fn=tools.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                  collate_fn=tools.collate_fn)

    # get model
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3)
    
    num_epochs =10
    for epoch in range(num_epochs):
        train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=10)
        lr_scheduler.step()
        # evaluate(model,data_loader_test,device=device)
    torch.save(model.state_dict(),"masknet.pth")
    print("OK!")

def predict(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_trans = transform(img)
    img_list = [img_trans]
    # print(img_input.size())
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load("masknet.pth"))
    model.eval()
    with torch.no_grad():
        output = model(img_list)

    return img, output[0]

def show_objects(img_path):
    img, result = predict(img_path)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()
    boxes = result['boxes'].numpy()
    labels = result['labels'].numpy()
    scores = result['scores'].numpy()
    masks = result['masks']
    mask_add = torch.zeros_like(masks[0][0])
    for i in range(len(boxes)):
        x_min,y_min,x_max,y_max = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        width, height = x_max-x_min, y_max-y_min
        rect = plt.Rectangle((x_min,y_min),width,height,edgecolor='red',
                             linewidth=1,fill=False)
        ax.text(x_min,y_max+20, str(CLASSES[labels[i]])+":"+str(round(scores[i],3)),color='green')
        ax.add_patch(rect)
        mask = torch.squeeze(masks[i])
        mask_add = torch.add(mask_add,mask)

    plt.figure()
    plt.imshow(mask_add.numpy())
    plt.axis('off')
    
    plt.show()
    



if __name__ == '__main__':
    # train(data_root='/home/librah/workspace/video_dataset/PennFudanPed')
    show_objects(img_path='/home/librah/workspace/video_dataset/PennFudanPed/1.jpg')

import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

from models.fcn import FCN
from utils.data import VOC2012, CenterCropWithIgnore
from utils.metric import compute_meanIU, compute_confusion

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = FCN()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)

dataset = VOC2012('./VOC2012',
                  input_transform=Compose([
                      CenterCrop(512),
                      ToTensor(),
                      Normalize([.485, .456, .406], [.229, .224, .225]),
                  ]), target_transform=CenterCropWithIgnore(512, 255))
loader = DataLoader(dataset, num_workers=1, batch_size=10, shuffle=True)

num_classes = 21


def train(model, loader, optimizer):
    model.to(device)
    model.train()

    all_loss = 0
    all_conf = torch.zeros(num_classes, num_classes)
    for i, data in enumerate(loader):
        image, label = Variable(data[0].to(device)), Variable(data[1].squeeze().to(device))

        optimizer.zero_grad()
        pred = model.forward(image)
        loss = loss_fn(pred, label[:, 0])
        loss.backward()
        optimizer.step()

        val, pred_seg = pred.cpu().max(1)
        all_loss += loss.item()
        all_conf = all_conf + compute_confusion(pred_seg, label.cpu())
        meaniu = compute_meanIU(all_conf)

        if (i % 10 == 0) or (i == len(loader) - 1):
            print("[{:4d}/{:4d}] loss:{:.3f} meanIU:{:.3f}".format(i, len(loader), all_loss / (i + 1), meaniu))


def test(model, loader):
    model.to(device)
    model.eval()

    all_loss = 0
    all_conf = torch.zeros(num_classes, num_classes)
    for i, data in enumerate(loader):
        image, label = Variable(data[0].to(device)), Variable(data[1].squeeze().to(device))

        pred = model.forward(image)
        loss = loss_fn(pred, label)

        val, pred_seg = pred.cpu().max(1)
        all_loss += loss.item()
        all_conf = all_conf + compute_confusion(pred_seg, label.cpu())
        meaniu = compute_meanIU(all_conf)

        if (i % 50 == 0) or (i == len(loader) - 1):
            print("[{:4d}/{:4d}] loss:{:.3f} meanIU:{:.3f}".format(i, len(loader), all_loss / (i + 1), meaniu))



train(model, loader, optimizer)

test(model, loader)

import matplotlib.pyplot as plt


def get_voc_colormap():
    N = 256
    VOCcolormap = np.zeros([N, 3], dtype=np.uint8)
    for i in range(0, N):
        (r,b,g,idx)=(0,0,0,i)
        for j in range(0, 8):
            r = r | ((idx & 1) << (7 - j))
            g = g | ((idx & 2) << (7 - j))
            b = b | ((idx & 4) << (7 - j))
            idx = idx >> 3
        VOCcolormap[i, :] = [r, g >> 1, b >> 2]
    return VOCcolormap

def return_pascal_segmentation(input_im):
    VOCcolormap = get_voc_colormap()
    im = Image.fromarray(input_im, mode='P')
    im.putpalette(np.reshape(VOCcolormap, 768, 'C'))
    return im

def display(pred):
    im_idx = 0
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(data[0][im_idx,:,:,:].permute(1,2,0)) # C x H x W --> H x W x C
    plt.title('input image with mean subtracted')

    plt.subplot(1,3,2)
    plt.imshow(return_pascal_segmentation(data[1][im_idx,0,:,:].numpy().astype(np.uint8)))
    plt.title('ground truth segmentation')

    plt.subplot(1,3,3)
    val, pred_seg = pred.cpu().max(1)
    plt.imshow(return_pascal_segmentation(pred_seg[im_idx].numpy().astype(np.uint8)))
    plt.title('predicted segmentation')

def test(model, loader):
    model.to(device)
    model.eval()

    all_loss = 0
    all_conf = torch.zeros(num_classes, num_classes)
    for i, data in enumerate(loader):
        image, label = data[0].to(device), data[1].squeeze().to(device)

        pred = model.forward(image)
        loss = loss_fn(pred, label)
        val, pred_seg = pred.cpu().max(1)

        all_loss += loss.item()
        all_conf = all_conf + compute_confusion(pred_seg, label.cpu())
        meaniu = compute_meanIU(all_conf)

        display(pred)
        break

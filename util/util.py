from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def tensor2ctim(input_image, imtype=np.uint16):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = (image_numpy + 1) / 2.0 * 65535.0
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_ct_image(image_numpy, image_path):
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    plt.savefig(image_path)

def save_ctA_image(image_numpy, image_path):
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    currentAxis = plt.gca()
    rect0 = patches.Rectangle((15, 50), 25, 40, linewidth=1, edgecolor='r', facecolor='none')
    rect1 = patches.Rectangle((18, 112), 18, 20, linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((50, 230), 30, 20, linewidth=1, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rect0)
    currentAxis.add_patch(rect1)
    currentAxis.add_patch(rect2)
    plt.savefig(image_path)

    mean_str = [str(np.mean(image_numpy[50:50+40, 15:15+25])), str(np.mean(image_numpy[112:112+20, 18:18+18])), str(np.mean(image_numpy[230:230+20, 50:50+30]))]
    std_str = [str(np.std(image_numpy[50:50+40, 15:15+25])), str(np.std(image_numpy[112:112+20, 18:18+18])), str(np.std(image_numpy[230:230+20, 50:50+30]))]

    return mean_str, std_str

def save_ctB_image(image_numpy, image_path):
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    currentAxis = plt.gca()
    rect0 = patches.Rectangle((60, 20), 40, 20, linewidth=1, edgecolor='r', facecolor='none')
    rect1 = patches.Rectangle((65, 70), 30, 15, linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((225, 160), 25, 10, linewidth=1, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rect0)
    currentAxis.add_patch(rect1)
    currentAxis.add_patch(rect2)
    plt.savefig(image_path)

    mean_str = [str(np.mean(image_numpy[20:20+20, 60:60+40])), str(np.mean(image_numpy[70:70+15, 65:65+30])), str(np.mean(image_numpy[160:160+10, 225:225+25]))]
    std_str = [str(np.std(image_numpy[20:20+20, 60:60+40])), str(np.std(image_numpy[70:70+15, 65:65+30])), str(np.std(image_numpy[160:160+10, 225:225+25]))]

    return mean_str, std_str

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

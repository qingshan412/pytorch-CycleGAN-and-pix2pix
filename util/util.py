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
    image_numpy = (image_numpy + 1) / 2.0 * 4095.0#65535.0
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

def save_ct_image(CTname, RESlabel, im_numpy, image_path):
    if CTname.split('+')[0] != "fbp":
        if CTname.split('_')[0] != "199" and CTname.split('_')[0] != "200":
            mean_str, std_str = save_ct_image_naive(im_numpy, image_path)
        else:
            mean_str, std_str = save_ctABo_image(im_numpy, image_path)
    else:
        if RESlabel in ['real_A', 'fake_B', 'rec_A', 'idt_B', 'fake_B_A', 'rec_A_B', 'fake_C_A', 'rec_A_C']:
            mean_str, std_str = save_ctA_image(im_numpy, image_path)
        else:
            mean_str, std_str = save_ctB_image(im_numpy, image_path)
    return mean_str, std_str

def save_ct_image_naive(image_numpy, image_path):
    plt.clf()
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    # plt.axis('off')
    currentAxisA = plt.gca()
    currentAxisA.axes.get_xaxis().set_visible(False)
    currentAxisA.axes.get_yaxis().set_visible(False)
    currentAxisA.spines['left'].set_color('none')
    currentAxisA.spines['bottom'].set_color('none')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)

    mean_str = ' '
    std_str = ' '

    return mean_str, std_str

def save_ctA_image(image_numpy, image_path):
    # print('A')
    plt.clf()
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    currentAxisA = plt.gca()
    rectA0 = patches.Rectangle((15, 50), 25, 40, linewidth=1, edgecolor='r', facecolor='none')
    rectA1 = patches.Rectangle((18, 112), 18, 20, linewidth=1, edgecolor='r', facecolor='none')
    rectA2 = patches.Rectangle((50, 230), 30, 20, linewidth=1, edgecolor='r', facecolor='none')
    currentAxisA.add_patch(rectA0)
    currentAxisA.add_patch(rectA1)
    currentAxisA.add_patch(rectA2)
    # plt.axis('off')
    currentAxisA.axes.get_xaxis().set_visible(False)
    currentAxisA.axes.get_yaxis().set_visible(False)
    currentAxisA.spines['left'].set_color('none')
    currentAxisA.spines['bottom'].set_color('none')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)

    mean_str = [str(round(np.mean(image_numpy[50:50+40, 15:15+25]),2)), str(round(np.mean(image_numpy[112:112+20, 18:18+18]),2)), str(round(np.mean(image_numpy[230:230+20, 50:50+30]),2))]
    std_str = [str(round(np.std(image_numpy[50:50+40, 15:15+25]),2)), str(round(np.std(image_numpy[112:112+20, 18:18+18]),2)), str(round(np.std(image_numpy[230:230+20, 50:50+30]),2))]

    return mean_str, std_str

def save_ctABo_image(image_numpy, image_path):
    # print('A')
    plt.clf()
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    currentAxisA = plt.gca()
    rectA0 = patches.Rectangle((15, 50), 20, 25, linewidth=1, edgecolor='r', facecolor='none')
    rectA1 = patches.Rectangle((25, 112), 18, 20, linewidth=1, edgecolor='r', facecolor='none')
    rectA2 = patches.Rectangle((100, 8), 25, 25, linewidth=1, edgecolor='r', facecolor='none')
    currentAxisA.add_patch(rectA0)
    currentAxisA.add_patch(rectA1)
    currentAxisA.add_patch(rectA2)
    # plt.axis('off')
    currentAxisA.axes.get_xaxis().set_visible(False)
    currentAxisA.axes.get_yaxis().set_visible(False)
    currentAxisA.spines['left'].set_color('none')
    currentAxisA.spines['bottom'].set_color('none')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)

    mean_str = [str(round(np.mean(image_numpy[50:50+25, 15:15+20]),2)), str(round(np.mean(image_numpy[112:112+20, 25:25+18]),2)), str(round(np.mean(image_numpy[8:8+25, 100:100+25]),2))]
    std_str = [str(round(np.std(image_numpy[50:50+25, 15:15+20]),2)), str(round(np.std(image_numpy[112:112+20, 25:25+18]),2)), str(round(np.std(image_numpy[8:8+25, 100:100+25]),2))]

    return mean_str, std_str

def save_ctB_image(image_numpy, image_path):
    # print('B')
    plt.clf()
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    currentAxisB = plt.gca()
    rectB0 = patches.Rectangle((60, 20), 40, 20, linewidth=1, edgecolor='r', facecolor='none')
    rectB1 = patches.Rectangle((65, 70), 30, 15, linewidth=1, edgecolor='r', facecolor='none')
    rectB2 = patches.Rectangle((225, 160), 25, 10, linewidth=1, edgecolor='r', facecolor='none')
    currentAxisB.add_patch(rectB0)
    currentAxisB.add_patch(rectB1)
    currentAxisB.add_patch(rectB2)
    # plt.axis('off')
    currentAxisB.axes.get_xaxis().set_visible(False)
    currentAxisB.axes.get_yaxis().set_visible(False)
    currentAxisB.spines['left'].set_color('none')
    currentAxisB.spines['bottom'].set_color('none')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)

    mean_str = [str(round(np.mean(image_numpy[20:20+20, 60:60+40]),2)), str(round(np.mean(image_numpy[70:70+15, 65:65+30]),2)), str(round(np.mean(image_numpy[160:160+10, 225:225+25]),2))]
    std_str = [str(round(np.std(image_numpy[20:20+20, 60:60+40]),2)), str(round(np.std(image_numpy[70:70+15, 65:65+30]),2)), str(round(np.std(image_numpy[160:160+10, 225:225+25]),2))]

    return mean_str, std_str


def save_merged_ct_image_8(image_numpy, image_path):
    plt.clf()
    plt.imshow(np.squeeze(image_numpy), cmap=plt.cm.bone)
    currentAxisA = plt.gca()
    if 'CT1_8_11703352_L2_HBIHI4DP_0IT0QIRM_I225_000' in image_path:
        rectA0 = patches.Rectangle((248, 285), 20, 18, linewidth=1, edgecolor='r', facecolor='none')
        rectA1 = patches.Rectangle((280, 160), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
        mean_str = [str(round(np.mean(image_numpy[285:285+18, 248:248+20]),2)), str(round(np.mean(image_numpy[160:160+40, 280:280+40]),2))]
        std_str = [str(round(np.std(image_numpy[285:285+18, 248:248+20]),2)), str(round(np.std(image_numpy[160:160+40, 280:280+40]),2))]
    if 'CT1_3_11712989_L2_5BWN3IGL_1F2AMF44_I12__000' in image_path:
        rectA0 = patches.Rectangle((48, 265), 16, 22, linewidth=1, edgecolor='r', facecolor='none')
        rectA1 = patches.Rectangle((110, 360), 25, 18, linewidth=1, edgecolor='r', facecolor='none')
        mean_str = [str(round(np.mean(image_numpy[265:265+22, 48:48+16]),2)), str(round(np.mean(image_numpy[360:360+18, 110:110+25]),2))]
        std_str = [str(round(np.std(image_numpy[265:265+22, 48:48+16]),2)), str(round(np.std(image_numpy[360:360+18, 110:110+25]),2))]
    if 'low_dose_data_10686825_FBP_JFO5YMQT_NSHOW2LY_I6000001' in image_path:
        rectA0 = patches.Rectangle((208, 187), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
        rectA1 = patches.Rectangle((160, 415), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
        mean_str = [str(round(np.mean(image_numpy[187:187+20, 208:208+20]),2)), str(round(np.mean(image_numpy[415:415+20, 160:160+20]),2))]
        std_str = [str(round(np.std(image_numpy[187:187+20, 208:208+20]),2)), str(round(np.std(image_numpy[415:415+20, 160:160+20]),2))]        
    if 'low_dose_data_1_10674581_FBP_VWBZW31O_5LSMFNNE_I2970000' in image_path:
        rectA0 = patches.Rectangle((55, 245), 30, 25, linewidth=1, edgecolor='r', facecolor='none')
        rectA1 = patches.Rectangle((250, 280), 30, 30, linewidth=1, edgecolor='r', facecolor='none')
        mean_str = [str(round(np.mean(image_numpy[245:245+25, 55:55+30]),2)), str(round(np.mean(image_numpy[280:280+30, 250:250+30]),2))]
        std_str = [str(round(np.std(image_numpy[245:245+25, 55:55+30]),2)), str(round(np.std(image_numpy[280:280+30, 250:250+30]),2))]
    if 'low_dose_data_1_10674581_FBP_VWBZW31O_5LSMFNNE_I5500000' in image_path:
        rectA0 = patches.Rectangle((90, 365), 30, 30, linewidth=1, edgecolor='r', facecolor='none')
        rectA1 = patches.Rectangle((430, 260), 30, 30, linewidth=1, edgecolor='r', facecolor='none')
        mean_str = [str(round(np.mean(image_numpy[365:365+30, 90:90+30]),2)), str(round(np.mean(image_numpy[260:260+30, 430:430+30]),2))]
        std_str = [str(round(np.std(image_numpy[365:365+30, 90:90+30]),2)), str(round(np.std(image_numpy[260:260+30, 430:430+30]),2))]
    
    currentAxisA.add_patch(rectA0)
    currentAxisA.add_patch(rectA1)

    currentAxisA.axes.get_xaxis().set_visible(False)
    currentAxisA.axes.get_yaxis().set_visible(False)
    currentAxisA.spines['left'].set_color('none')
    currentAxisA.spines['bottom'].set_color('none')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.0)

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

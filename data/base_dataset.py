import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

import random, numbers
import torch
import numpy as np


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __adjust(img)))
    elif opt.resize_or_crop == 'ct_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __ct_random_crop(img, opt.fineSize)))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    if opt.resize_or_crop == 'ct_crop':
        if opt.isTrain and not opt.no_flip:
            transform_list.append(transforms.Lambda(
                lambda img: __ct_random_flip(img)))

        transform_list += [transforms.Lambda(
                lambda img: __ct_to_tensor(img)),
                        transforms.Lambda(
                lambda img: __ct_normalize(img))]
    # elif opt.resize_or_crop == 'one_crop':
    #     if opt.isTrain and not opt.no_flip:
    #         transform_list.append(transforms.Lambda(
    #             lambda img: __ct_random_flip(img)))

    #     transform_list += [transforms.Lambda(
    #             lambda img: __ct_to_tensor(img)),]
    #                        transforms.Lambda(
    #             lambda img: __ct_normalize(img))]
    else:
        if opt.resize_or_crop != 'none' and opt.isTrain and not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


# just modify the width and height to be multiple of 4
def __adjust(img):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

def __ct_random_crop(img, target_size):
    # print(np.max(img))
    if np.max(img) > 2:
        img = img/4095.#65535. -> [0, 1]
    elif np.max(img) > 1:
        # print('here...')
        img = img/2.
    ow, oh = img.shape
    
    ### input is supposed to be a float32 [0,1] numpy array here
    ### output should be a tensor in pytorch

    ### crop
    if isinstance(target_size, numbers.Number):
        tw, th = int(target_size), int(target_size)
    else:
        tw, th = target_size

    i = random.randint(0, ow - tw)
    j = random.randint(0, oh - th)

    res = img[i:i+tw, j:j+th] 
    
    return res

def __ct_random_flip(img):
    res = img.copy()
    ### flip
    if random.randint(0, 1) < 1:
        res = np.flip(res, 0).copy() ### vertical
    if random.randint(0, 1) > 0:
        res = np.flip(res, 1).copy() ## horizontal
    
    return res

def __ct_to_tensor(res):
    ### to tensor
    return torch.from_numpy(res)

def __ct_normalize(res):
    ### normalize
    return (res - 0.5)/0.5
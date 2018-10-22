import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_ct_dataset
from PIL import Image
import random

# import pydicom
import numpy as np


class UnalignedCTDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_ct_dataset(self.dir_A) ### [image_dir_0, image_dir_1, image_dir_2,...]
        self.B_paths = make_ct_dataset(self.dir_B) ### [image_dir_0, image_dir_1, image_dir_2,...]

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        ### uint16 to float32 to [0, 1]
        # A_img = pydicom.dcmread(A_path).pixel_array.astype(np.float32)/65535.
        # B_img = pydicom.dcmread(B_path).pixel_array.astype(np.float32)/65535.
        A_img = np.load(A_path).astype(np.float32)#/65535.
        B_img = np.load(B_path).astype(np.float32)#/65535.

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.direction == 'BtoA': ### one direction???
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
        #     tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = A.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = B.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

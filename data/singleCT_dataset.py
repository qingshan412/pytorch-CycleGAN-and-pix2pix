import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_ct_dataset
# from PIL import Image

import numpy as np

class SingleCTDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')

        self.A_paths = make_ct_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = np.load(A_path).astype(np.float32)
        # Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            # tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            # A = tmp.unsqueeze(0)
            A = A.unsqueeze(0)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleCTDataset'

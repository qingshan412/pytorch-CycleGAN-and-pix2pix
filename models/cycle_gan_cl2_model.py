import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANcl2Model(BaseModel):
    def name(self):
        return 'CycleGANcl2Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> C -> A) and (A -> C -> B -> C -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> C -> B) and (B -> C -> A -> C -> B)')
            parser.add_argument('--lambda_C', type=float, default=10.0, help='weight for cycle loss (C -> A -> C) and (C -> B -> C)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A_B', 'G_A', 'cycle_A', 'idt_A', 'D_B_A', 'G_B', 'cycle_B', 'idt_B', 'D_C_A', 'D_C_B', 'G_C', 'idt_C']
        # 'D_A_C', 'D_B_C', , 'cycle_C'
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_C_A', 'fake_B_A', 'rec_A_B']#, 'rec_A_C']
        visual_names_B = ['real_B', 'fake_C_B', 'fake_A_B', 'rec_B_A']#, 'rec_B_C']
        visual_names_C = ['real_C', ]#'fake_A_C', 'fake_B_C', 'rec_C_A', 'rec_C_B']#
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
            visual_names_C.append('idt_C_A')
            visual_names_C.append('idt_C_B')

        self.visual_names = visual_names_A + visual_names_B + visual_names_C
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A_C', 'G_B_C', 'D_A', 'D_B', 'G_C_A', 'G_C_B', 'D_C']
        else:  # during test time, only load Gs
            self.model_names = ['G_A_C', 'G_B_C', 'G_C_A', 'G_C_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_C_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B_C = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.netG_C_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A_C = networks.define_G(opt.input_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            
            self.netD_C = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_B_pool = ImagePool(opt.pool_size)
            # self.fake_A_C_pool = ImagePool(opt.pool_size)

            self.fake_B_A_pool = ImagePool(opt.pool_size)
            # self.fake_B_C_pool = ImagePool(opt.pool_size)

            self.fake_C_A_pool = ImagePool(opt.pool_size)
            self.fake_C_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A_C.parameters(), self.netG_B_C.parameters(), 
                                                self.netG_C_A.parameters(), self.netG_C_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), 
                                                self.netD_C.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_C_A = self.netG_C_A(self.real_A)
        # self.rec_A_C = self.netG_A_C(self.fake_C_A)
        # self.fake_A_C = self.netG_A_C(self.real_C)
        # self.rec_C_A = self.netG_C_A(self.fake_A_C)

        # self.real_C.detach_()

        self.fake_C_B = self.netG_C_B(self.real_B)
        # self.rec_B_C = self.netG_B_C(self.fake_C_B)
        # self.fake_B_C = self.netG_B_C(self.real_C)
        # self.rec_C_B  = self.netG_C_B(self.fake_B_C)

        # self.fake_B = self.netG_A(self.real_A)
        # self.rec_A = self.netG_B(self.fake_B)

        # self.fake_A = self.netG_B(self.real_B)
        # self.rec_B = self.netG_A(self.fake_A)
        
        self.fake_B_A = self.netG_B_C(self.fake_C_A)
        self.rec_A_B = self.netG_A_C(self.netG_C_B(self.fake_B_A))
        self.fake_A_B = self.netG_A_C(self.fake_C_B)
        self.rec_B_A = self.netG_B_C(self.netG_C_A(self.fake_A_B))

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        # fake_A_C = self.fake_A_C_pool.query(self.fake_A_C)
        # self.loss_D_A_C = self.backward_D_basic(self.netD_A, self.real_A, fake_A_C)

        fake_A_B = self.fake_A_B_pool.query(self.fake_A_B)
        self.loss_D_A_B = self.backward_D_basic(self.netD_A, self.real_A, fake_A_B)

    def backward_D_B(self):
        # fake_B_C = self.fake_B_C_pool.query(self.fake_B_C)
        # self.loss_D_B_C = self.backward_D_basic(self.netD_B, self.real_B, fake_B_C)

        fake_B_A = self.fake_B_A_pool.query(self.fake_B_A)
        self.loss_D_B_A = self.backward_D_basic(self.netD_B, self.real_B, fake_B_A)

    def backward_D_C(self):
        fake_C_A = self.fake_C_A_pool.query(self.fake_C_A)
        self.loss_D_C_A = self.backward_D_basic(self.netD_C, self.real_C, fake_C_A) 

        fake_C_B = self.fake_C_B_pool.query(self.fake_C_B)
        self.loss_D_C_B = self.backward_D_basic(self.netD_C, self.real_C, fake_C_B) 

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_C_A(self.real_C)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_C) * lambda_C * lambda_idt * 2.
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_C_B(self.real_C)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_C) * lambda_C * lambda_idt * 2.
            # G_A/B_C should be identity if real_A/B is fed.
            self.idt_C_A = self.netG_A_C(self.real_A)
            # self.loss_idt_C_A = self.criterionIdt(self.idt_C_A, self.real_A) * lambda_A * lambda_idt / 2.
            self.idt_C_B = self.netG_C_B(self.real_B)
            # self.loss_idt_C_B = self.criterionIdt(self.idt_C_B, self.real_B) * lambda_A * lambda_idt / 2.
            self.loss_idt_C = self.criterionIdt(self.idt_C_A, self.real_A) * lambda_A * lambda_idt + self.criterionIdt(self.idt_C_B, self.real_B) * lambda_A * lambda_idt#)/ 2.
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            self.loss_idt_C = 0

        self.loss_idt = self.loss_idt_A + self.loss_idt_B + self.loss_idt_C#_A + self.loss_idt_C_B

        # GAN loss D_A(G_A(A)) different from original code D_A for B and D_B for A, I use D_A for A, D_B for B, and D_C for C
        self.loss_G_C = self.criterionGAN(self.netD_C(self.fake_C_A), True) + self.criterionGAN(self.netD_C(self.fake_C_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_A_B), True) * 2. # + self.criterionGAN(self.netD_A(self.fake_A_C), True)
        # GAN loss D_C(G_C_A/B(C_A/B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B_A), True) * 2. # + self.criterionGAN(self.netD_B(self.fake_B_C), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A_B, self.real_A) * lambda_A * 2. # + self.criterionCycle(self.rec_A_C, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B_A, self.real_B) * lambda_B * 2. # + self.criterionCycle(self.rec_B_C, self.real_B) * lambda_B
        # Backward cycle loss
        # self.loss_cycle_C = self.criterionCycle(self.rec_C_A, self.real_C) * lambda_C + self.criterionCycle(self.rec_C_B, self.real_C) * lambda_C
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_C + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt# + self.loss_cycle_C
        self.loss_G.backward()
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed.
        #     self.idt_A = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #     # G_B should be identity if real_A is fed.
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0

        # # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # # Forward cycle loss
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # # Backward cycle loss
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # # combined loss
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.backward_D_C()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

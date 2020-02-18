
import os
import torch
import torch.nn as nn
import itertools

from torch.nn import init
from collections import OrderedDict
from networks import Generator, Discriminator

from utils.image_pool import ImagePool

class Cyclegan():
    """
    This class defines the cyclegan model, loss functions and optimizers.
    A is the source domain and B is the target domain. 

    Generators:
        self.netG_A: generates fake_B by self.netG_A(real_A)
        self.netG_B: generates fake_A by self.netG_B(real_B)

    Descriminators:
        self.netD_A: discriminates between fake_B and real_B
        self.netD_B: discriminates between fake_A and real_A
    """

    def __init__(self, device):
        self.device = device        
        self.netG_A = self.__init_weights(Generator(3, use_dropout=False).to(self.device))
        self.netG_B = self.__init_weights(Generator(3, use_dropout=False).to(self.device))
        self.netD_A = self.__init_weights(Discriminator(3).to(self.device))
        self.netD_B = self.__init_weights(Discriminator(3).to(self.device))
        
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_idt = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(
            self.netG_A.parameters(), self.netG_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(
            self.netD_A.parameters(), self.netD_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        self.lambda_A = 10
        self.lambda_B = 10
        self.lambda_idt = 0.5

        self.save_dir = './models'

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def __init_weights(self, net, init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal_(m.weight.data, 0.0, init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        net.apply(init_func)
        return net

    def __optimize_G(self, real_A, real_B):
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()

        def generator_loss(real_source, target_generator, target_discriminator):
            # generator should generate a fake image which can fool discriminator
            fake_target = target_generator(real_source)
            prediction_of_fake_target = target_discriminator(fake_target)
            loss = self.criterion_gan(
                prediction_of_fake_target, torch.tensor(1.0).to(self.device).expand_as(prediction_of_fake_target))
            return loss

        def cycle_loss(real_source, target_generator, source_generator):
            fake_target = target_generator(real_source)
            reconstructed_source = source_generator(fake_target)
            loss = self.criterion_cycle(reconstructed_source, real_source)
            return loss

        def identity_loss(real_source, target_generator):
            fake_target = target_generator(real_source)
            loss = self.criterion_idt(fake_target, real_source)
            return loss

        self.loss_G_A = generator_loss(real_A, self.netG_A, self.netD_A)
        self.loss_G_B = generator_loss(real_B, self.netG_B, self.netD_B)
        self.loss_cycle_A = cycle_loss(real_A, self.netG_A, self.netG_B) * self.lambda_A 
        self.loss_cycle_B = cycle_loss(real_B, self.netG_B, self.netG_A) * self.lambda_B
        self.loss_idt_A = identity_loss(real_B, self.netG_A) * self.lambda_B * self.lambda_idt
        self.loss_idt_B = identity_loss(real_A, self.netG_B) * self.lambda_A * self.lambda_idt

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

        self.optimizer_G.step()

    def __optimize_D(self, real_A, real_B):
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()

        def discriminator_loss(real_source, real_target, fake_target_pool, target_generator, target_discriminator):
            # discriminator should predict fake_target as False because fake_target is not a real image
            fake_target = target_generator(real_source)
            fake_target = fake_target_pool.query(fake_target)
            prediction_of_fake_target = target_discriminator(
                fake_target.detach())
            loss_fake = self.criterion_gan(
                prediction_of_fake_target, torch.tensor(0.0).to(self.device).expand_as(prediction_of_fake_target))

            # Also, discriminator should predict real_target as True because real_target is a real image
            prediction_of_real_target = target_discriminator(real_target)
            loss_real = self.criterion_gan(
                prediction_of_real_target, torch.tensor(1.0).to(self.device).expand_as(prediction_of_real_target))

            loss = (loss_fake + loss_real) * 0.5
            return loss

        # optimize netD_A
        self.loss_D_A = discriminator_loss(real_A, real_B, self.fake_B_pool, self.netG_A, self.netD_A)
        self.loss_D_A.backward()

        # optimize netD_B
        self.loss_D_B = discriminator_loss(real_B, real_A, self.fake_A_pool, self.netG_B, self.netD_B)
        self.loss_D_B.backward()

        self.optimizer_D.step()

    def optimize_parameters(self, real_A, real_B):
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        self.__optimize_G(real_A, real_B)
        self.__optimize_D(real_A, real_B)

    def forward(self, real_A, real_B):
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        return self.netG_A(real_A), self.netG_B(real_B)

    def save_networks(self, epoch):    
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)        

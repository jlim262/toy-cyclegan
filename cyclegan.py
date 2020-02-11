
import torch
import torch.nn as nn
import itertools

from collections import OrderedDict
from networks import Generator, Discriminator


class Cyclegan():
    def __init__(self):
        self.netG_A = Generator(3)
        self.netG_B = Generator(3)
        self.netD_A = Discriminator(3)
        self.netD_B = Discriminator(3)

        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_idt = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(
            self.netG_A.parameters(), self.netG_B.parameters()), lr=0.2, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(
            self.netD_A.parameters(), self.netD_B.parameters()), lr=0.2, betas=(0.5, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']

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

    def __optimize_G(self, real_A, real_B):
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()

        def generator_loss(real, generator, discriminator):
            fake = generator(real)
            pred_fake = discriminator(fake.detach())
            loss = self.criterion_gan(
                pred_fake, torch.tensor(1.0).expand_as(pred_fake))
            return loss

        def cycle_loss(real, generator1, generator2):
            fake = generator1(real)
            reconstructed = generator2(fake.detach())
            loss = self.criterion_cycle(reconstructed, real)
            return loss

        self.loss_G_A = generator_loss(real_A, self.netG_A, self.netD_A)
        self.loss_G_B = generator_loss(real_B, self.netG_B, self.netD_B)
        self.loss_cycle_A = cycle_loss(real_A, self.netG_A, self.netG_B)
        self.loss_cycle_B = cycle_loss(real_B, self.netG_B, self.netG_A)

        self.loss_G = self.loss_G_A + self.loss_G_B + \
            self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

        self.optimizer_G.step()

    def __optimize_D(self, real_A, real_B):
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()

        def discriminator_loss(real, generator, discriminator):
            fake = generator(real)
            pred_fake = discriminator(fake.detach())
            loss_fake = self.criterion_gan(
                pred_fake, torch.tensor(0.0).expand_as(pred_fake))

            pred_real = discriminator(real)
            loss_real = self.criterion_gan(
                pred_real, torch.tensor(1.0).expand_as(pred_real))

            loss = loss_fake + loss_real
            return loss

        # optimize netD_A
        self.loss_D_A = discriminator_loss(real_B, self.netG_A, self.netD_A)
        self.loss_D_A.backward()

        # optimize netD_B
        self.loss_D_B = discriminator_loss(real_A, self.netG_B, self.netD_B)
        self.loss_D_B.backward()

        self.optimizer_D.step()

    def optimize_parameters(self, real_A, real_B):
        self.__optimize_G(real_A, real_B)
        self.__optimize_D(real_A, real_B)

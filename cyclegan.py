
import torch
import torch.nn as nn
import itertools

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
        
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=0.2, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=0.2, betas=(0.5, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]

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
            loss = self.criterion_gan(pred_fake, True)
            return loss

        def cycle_loss(real, generator1, generator2):
            fake = generator1(real)
            reconstructed = generator2(fake.detach())
            loss = self.criterion_cycle(reconstructed, real)
            return loss

        loss_G_A = generator_loss(real_A, self.netG_A, self.netD_A)
        loss_G_B = generator_loss(real_B, self.netG_B, self.netD_B)
        loss_cycle_A = cycle_loss(real_A, self.netG_A, self.netG_B)
        loss_cycle_B = cycle_loss(real_B, self.netG_B, self.netG_A)
        
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        loss_G.backward()
        
        self.optimizer_G.step()

    def __optimize_D(self, real_A, real_B):
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()

        def optimize(real, generator, discriminator):
            fake = generator(real)
            pred_fake = discriminator(fake.detach())
            loss_fake = self.criterion_gan(pred_fake, False)

            pred_real = discriminator(real)
            loss_real = self.criterion_gan(pred_real, True)            

            loss_D = loss_fake + loss_real
            loss_D.backward()

        # optimize netD_A
        optimize(real_B, self.netG_A, self.netD_A)

        # optimize netD_B
        optimize(real_A, self.netG_B, self.netD_B)

        self.optimizer_D.step()  

    def optimize_parameters(self, real_A, real_B):
        self.__optimize_G(real_A, real_B)
        self.__optimize_D(real_A, real_B)

import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data

from torch.utils.tensorboard import SummaryWriter
from cyclegan import Cyclegan
from image_dataset import ImageDataset
from PIL import Image
from utils import tensor2im


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Cyclegan(device)

    dataset = ImageDataset('./dataset/bart2lisa')
    dataloader = data.DataLoader(dataset)

    writer = SummaryWriter()

    for epoch in range(40):
        print('epoch {}'.format(epoch))
        for i, d in enumerate(dataloader):
            model.optimize_parameters(d['A'], d['B'])

        test_it = iter(dataloader)
        d = next(test_it)
        test_A, test_B = d['A'], d['B']

        fake_B, fake_A = model.forward(test_A, test_B)
        writer.add_image('real_A', np.transpose(tensor2im(test_A), (2, 0, 1)), epoch)
        writer.add_image('fake_B', np.transpose(tensor2im(fake_B), (2, 0, 1)), epoch)
        writer.add_image('real_B', np.transpose(tensor2im(test_B), (2, 0, 1)), epoch)
        writer.add_image('fake_A', np.transpose(tensor2im(fake_A), (2, 0, 1)), epoch)
        writer.add_scalars('current_loss', dict(model.get_current_losses()), epoch)

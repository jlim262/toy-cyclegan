
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from cyclegan import Cyclegan
from image_dataset import ImageDataset
from PIL import Image

from utils.util import tensor2im

def write_to_tensorboard(writer, real_A, real_B, fake_A, fake_B, step):
    img_grid_AB = make_grid([real_A.squeeze(0), fake_B.squeeze(0)])
    img_grid_BA = make_grid([real_B.squeeze(0), fake_A.squeeze(0)])
    writer.add_image('[test] real_A vs fake_B', np.transpose(tensor2im(img_grid_AB.unsqueeze(0)), (2, 0, 1)), step)
    writer.add_image('[test] real_B vs fake_A', np.transpose(tensor2im(img_grid_BA.unsqueeze(0)), (2, 0, 1)), step)    

if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Cyclegan(device)

    dataset = ImageDataset('./dataset/bart2lisa')
    dataloader = data.DataLoader(dataset)

    writer = SummaryWriter()

    model.load_networks(0)

    for i, d in enumerate(dataloader):
        real_A, real_B = d['A'], d['B']

        # visualize results on tensorboard
        if ((i % 100) == 0):
            fake_B, fake_A = model.forward(real_A, real_B)
            write_to_tensorboard(writer, real_A, real_B, fake_A.detach().cpu(), fake_B.detach().cpu(), i)
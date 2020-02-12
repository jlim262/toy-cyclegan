
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data

from torch.utils.tensorboard import SummaryWriter
from cyclegan import Cyclegan
from image_dataset import ImageDataset
from PIL import Image


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / \
            2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


if __name__ == '__main__':
    model = Cyclegan()
    dataset = ImageDataset('./dataset/bart2lisa')
    dataloader = data.DataLoader(dataset)

    writer = SummaryWriter()

    for epoch in range(10):
        test_img = None
        for i, d in enumerate(dataloader):
            model.optimize_parameters(d['A'], d['B'])        
            test_img = d['A']

        img = tensor2im(model.forward(test_img))
        writer.add_image('fake_B', np.transpose(img, (2, 0, 1)), epoch)
        writer.add_scalars('current_loss', dict(model.get_current_losses()), epoch)

    

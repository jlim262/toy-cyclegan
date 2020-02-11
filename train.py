
import torch.utils.data as data

from torch.utils.tensorboard import SummaryWriter
from cyclegan import Cyclegan
from image_dataset import ImageDataset

if __name__ == '__main__':
    model = Cyclegan()
    dataset = ImageDataset('./dataset/bart2lisa')
    dataloader = data.DataLoader(dataset)

    writer = SummaryWriter()

    for i, d in enumerate(dataloader):
        model.optimize_parameters(d['A'], d['B'])
        writer.add_scalars('current_loss', dict(model.get_current_losses()), i)

    print(dict(model.get_current_losses()))

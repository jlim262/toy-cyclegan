
import torch.utils.data as data

from cyclegan import Cyclegan
from image_dataset import ImageDataset

if __name__ == '__main__':
    model = Cyclegan()    
    dataset = ImageDataset('./dataset/bart2lisa')
    dataloader = data.DataLoader(dataset)    

    for d in dataloader:
        model.optimize_parameters(d['A'], d['B'])
        
    
        
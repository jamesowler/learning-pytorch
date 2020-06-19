from __future__ import print_function, division
import os
import platform
import glob
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transforms import Resize_zero_pad, Windsorise, RandomRotationAboutZ

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

'''
torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that dataset[i] can be used to get ith sample
'''

# Cross compatable path names
if platform.system() == 'Windows':
    _split = '\\'
else:
    _split = '/'

class CatsAndDogsDataset(Dataset):
    ''' Cats and Dogs dataset from Microsoft research'''

    def __init__(self, rootdir, transform=None, resize=(256,256)):
        self.resize = resize
        self.rootdir = rootdir
        self.transform = transform
        # create dictionary of classes with indexes
        classes = [x.split(_split)[-1] for x in glob.glob(rootdir + f'{_split}*')]
        self.classes_dict = {i: n for n, i in enumerate(classes)}
        self.images = glob.glob(rootdir + f'{_split}*{_split}*.jpg')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_class = image_path.split(_split)[-2]
        img = Image.open(image_path).convert('RGB')
        img = np.array(img.resize(self.resize, Image.BILINEAR))
        label = np.array(self.classes_dict[image_class])

        if self.transform:
            img = self.transform(img)

        return img, label

if __name__ == '__main__':
    # create dataset object - batch must contain either tensors, np ndarrys, numbers, dicts, or lists - in this case we are using ndarrys as our custom transforms make use of them
    dataset = CatsAndDogsDataset('./image_classification/cats-and-dogs/PetImages', 
    transform=transforms.Compose([Resize_zero_pad((256, 256), 1), RandomRotationAboutZ(60, order=1), transforms.ToTensor()]))
    # DataLoader - combine dataset and sampler - enables itteration over dataset
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        img, label = sample_batched
        print(img.shape)

        img = img.numpy()
        img = np.transpose(img, axes=[2, 3, 1, 0]).astype(int)
        plt.imshow(img[:, :, :, 0])
        plt.show()

        if i_batch > 30:
            break

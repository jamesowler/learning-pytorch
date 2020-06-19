import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from dataset_loader import CatsAndDogsDataset
from custom_transforms import Resize_zero_pad, Windsorise, RandomRotationAboutZ, Normalise

# Object for writing information to tensorboard
writer = SummaryWriter('./runs/experiment_1')

# GPU or CPU?
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# parameters
num_epochs = 10
batch_size = 32
lr = 0.001
model_save_path = './cat_or_dog.pth'

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input channels: RGB
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*128*128, 120)
        self.fc2 = nn.Linear(120, 84)
        # 2 output nodes - dog or cat?
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Instantiate network and move to GPU
net = Net()
net.to(device)

# Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(net.parameters(), lr=lr)

# Training
transforms_ = transforms.Compose([
    RandomRotationAboutZ(60, order=1),
    Normalise(),
    Resize_zero_pad((256, 256), 1),
    transforms.ToTensor()
])

dataset = CatsAndDogsDataset('./image_classification/cats-and-dogs/PetImages', transform= transforms_)
trainloader = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=8)
number_of_training_batches = len(trainloader)

def train():
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # load data and move to GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero parameter gradients
            optimiser.zero_grad()

            # forward + backward + optmise
            outputs = net(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # print status
            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss/(i+1), (epoch-1)*len(trainloader) + i+1)

            end = '' if i < number_of_training_batches - 1 else '\n'
            print('\rEpoch {} -- {}/{} -- loss: {:.4f}'.format(epoch, i, len(trainloader), running_loss/(i+1)), end = end)

    print('Finished Training')
    torch.save(net.state_dict(), model_save_path)

if __name__ == '__main__':
    print('Starting training...')
    train()
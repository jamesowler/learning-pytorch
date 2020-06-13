import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# Object for writing information to tensorboard
writer = SummaryWriter('./mnist_example/runs/experiment_1')

# GPU or CPU?
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load data
transform_comp = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
trainset = datasets.MNIST('./mnist_example', download=False, train=True, transform=transform_comp)
testset = datasets.MNIST('./mnist_example', download=False, train=False, transform=transform_comp)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)

# parameters
num_epochs = 10
lr = 0.001
model_save_path = './mnist_net.pth'

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*14*14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # print status
            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss/(i+1), (epoch-1)*len(trainloader) + i+1)

            end = '' if i < number_of_training_batches - 1 else '\n'
            print('\rEpoch {} -- loss: {:.4f}'.format(epoch, running_loss/(i+1)), end = end)

    print('Finished Training')
    torch.save(net.state_dict(), model_save_path)


def test():
    net = Net()
    net.load_state_dict(torch.load(model_save_path))
    example_output = net(trainloader.dataset[0][0][0][0])



if __name__ == '__main__':
    train()
    # test()
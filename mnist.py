from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import MLP
from utils import *

print('loading data ...')

# loading data
training_data = datasets.MNIST(root='data', download=True, train=True, transform=ToTensor())
test_data = datasets.MNIST(root='data', download=True, train=False, transform=ToTensor())

# create dataloaders
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

print('loading data complete\n')

print('preparing model ...')

# preparing device
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')
device = torch.device(device)


# preparing model
model = MLP().to(device)

print('model ready\n')

print('optimizing model ...')

# prepare optimization
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10

for i in range(epochs):
    print(f'epoch {i+1}\n----------------------')
    train(train_dataloader, model, loss_fn, optimizer, device)
    test_cl(test_dataloader, model, loss_fn, device)
    print()

print('optimization complete\n')

print('saving model ...')
torch.save(model.state_dict(), 'test_model.pth')
print('model saved\n')

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import CifarCNN
from utils import *
import logging

set_up_log('cifar_clf')

# set parameters
batch_size = 256
loss_fn = nn.CrossEntropyLoss()
lr = 0.001
epochs = 150

logging.info('loading data')

# loading data
training_data = datasets.CIFAR10(root='data.nosync', download=True, train=True, transform=ToTensor())
test_data = datasets.CIFAR10(root='data.nosync', download=True, train=False, transform=ToTensor())

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = 'mps' if torch.backends.mps.is_available() \
    else 'cuda' if torch.cuda.is_available() \
    else 'cpu'
logging.info(f'using {device} device')
device = torch.device(device)

# preparing model
model = CifarCNN(n_channel=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 120])

logging.info('model ready')
logging.info('optimising model')

# optimising model
for i in range(epochs):
    logging.info(f'epoch {i + 1}')
    train_cl(train_dataloader, model, loss_fn, optimizer, device)
    test_cl(test_dataloader, model, loss_fn, device)
    scheduler.step()

logging.info('optimisation complete')
logging.info('saving model')
torch.save(model.state_dict(), 'CifarCNN.pth')
logging.info('model saved')

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import VGG16
from utils import *
import logging

set_up_log('cifar_aec')

# set parameters
batch_size = 32
loss_fn = nn.MSELoss()
lr = 0.001
epochs = 400

logging.info('loading data')

# loading data
training_data = datasets.CIFAR10(root='data.nosync', download=True, train=True, transform=ToTensor())
training_classes = torch.tensor(training_data.targets)
training_ind = (training_classes == 0) | (training_classes == 1)
training_data = Subset(training_data, torch.nonzero(training_ind))

test_data = datasets.CIFAR10(root='data.nosync', download=True, train=False, transform=ToTensor())
test_classes = torch.tensor(test_data.targets)
test_ind = (test_classes == 0) | (test_classes == 1)
test_data = Subset(test_data, torch.nonzero(test_ind))

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
model = VGG16().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300])

logging.info('model ready')
logging.info('optimising model')

# optimising model
for i in range(epochs):
    logging.info(f'epoch {i + 1}')
    train_ae(train_dataloader, model, loss_fn, optimizer, device)
    if (i + 1) % 10 == 0:
        test_ae(train_dataloader, model, loss_fn, device, 'training')
        test_ae(test_dataloader, model, loss_fn, device)
    scheduler.step()

logging.info('optimisation complete')
logging.info('saving model')
torch.save(model.state_dict(), 'CifarAEC.pth')
logging.info('model saved')

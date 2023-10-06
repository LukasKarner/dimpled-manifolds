from torch.utils.data import DataLoader
from torchvision import datasets
from models import CifarCNN
from utils import *

set_up_log('cifar_clf')

# set parameters
batch_size = 200
loss_fn = nn.CrossEntropyLoss()
lr = 0.001
epochs = 150

logging.info('loading data')

# loading data
training_data = datasets.CIFAR10(
    root='data.nosync',
    download=True,
    train=True,
    transform=train_transform(32),
)
test_data = datasets.CIFAR10(
    root='data.nosync',
    download=True,
    train=False,
    transform=eval_transform(),
)

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = get_device()

# preparing model
model = CifarCNN(n_channel=128).to(device)
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

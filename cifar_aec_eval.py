from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from models import CifarAEC
from utils import *

log_to_stdout()

# set parameters
batch_size = 8

logging.info('loading data')

# loading data
training_data = datasets.CIFAR10(
    root='data.nosync',
    download=True,
    train=True,
    transform=ToTensor(),
)
training_classes = torch.tensor(training_data.targets)
training_ind = (training_classes == 0) | (training_classes == 1)
training_data = Subset(training_data, torch.nonzero(training_ind))

test_data = datasets.CIFAR10(
    root='data.nosync',
    download=True,
    train=False,
    transform=ToTensor(),
)
test_classes = torch.tensor(test_data.targets)
test_ind = (test_classes == 0) | (test_classes == 1)
test_data = Subset(test_data, torch.nonzero(test_ind))

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = torch.device('cpu')

# preparing model
model = CifarAEC().to(device)
model.load_state_dict(torch.load('temp/CifarAEC.pth', map_location=device), strict=True)
model.eval()

logging.info('model ready')

for X, labels in train_dataloader:
    break
X = X.to(device)
Y = model(X)

X, Y = X.to('cpu').detach(), torch.clamp(Y.to('cpu').detach(), 0., 1.)
aec_example_plot(X, Y, 'cifar_train')

for X, labels in test_dataloader:
    break
X = X.to(device)
Y = model(X)

X, Y = X.to('cpu').detach(), torch.clamp(Y.to('cpu').detach(), 0., 1.)
aec_example_plot(X, Y, 'cifar_test')

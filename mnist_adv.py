from torch.utils.data import DataLoader
from torchvision import datasets
from models import MnistMLP
from utils import *

log_to_stdout()

# set parameters
batch_size = 1
loss_fn = MarginLoss()

logging.info('loading data')

# loading data
# training_data = datasets.MNIST(root='data.nosync', download=True, train=True, transform=ToTensor())
test_data = datasets.MNIST(root='data.nosync', download=True, train=False, transform=ToTensor())

# create dataloaders
# train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = 'cpu'

# preparing model
model = MnistMLP().to(device)
model.load_state_dict(torch.load('finals/MnistMLP.pth', map_location=device), strict=True)
model.eval()

logging.info('model ready')
logging.info('running trials')



logging.info('trials complete')

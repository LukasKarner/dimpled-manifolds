from torch.utils.data import DataLoader
from torchvision import datasets
from models import MnistMLP
from utils import *

set_up_log('mnist_adv')

# set parameters
batch_size = 1
loss_fn = MarginLoss()

logging.info('loading data')

# loading data
training_data = datasets.MNIST(root='data.nosync', download=True, train=True, transform=eval_transform(1))
test_data = datasets.MNIST(root='data.nosync', download=True, train=False, transform=eval_transform(1))

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = 'cpu'

# preparing model
model = MnistMLP().to(device)
model.load_state_dict(torch.load('finals/MnistMLP.pth', map_location=device), strict=True)

logging.info('model ready')
logging.info('running trials')

examples = adv_attack_standard(model, train_dataloader, 2., 0.04, device, loss_fct=loss_fn)
adv_example_plot(examples, 'mnist_adv_train', transform=inv_scaling(1))

examples = adv_attack_standard(model, test_dataloader, 2., 0.04, device, loss_fct=loss_fn)
adv_example_plot(examples, 'mnist_adv_test', transform=inv_scaling(1))

logging.info('trials complete')

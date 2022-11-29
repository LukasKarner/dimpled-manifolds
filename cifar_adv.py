from torch.utils.data import DataLoader
from torchvision import datasets
from models import CifarCNN
from utils import *

set_up_log('cifar_adv')

# set parameters
batch_size = 1
loss_fn = MarginLoss()

logging.info('loading data')

# loading data
training_data = datasets.CIFAR10(
    root='data.nosync',
    download=True,
    train=True,
    transform=eval_transform(),
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
model.load_state_dict(torch.load('finals/CifarCNN.pth', map_location=device), strict=False)

logging.info('model ready')
logging.info('running trials')

examples = adv_attack_standard(model, train_dataloader, 0.5, 0.04, device, loss_fct=loss_fn)
adv_example_plot(examples, 'cifar_adv_train', transform=inv_scaling(), labels=training_data.classes)

examples = adv_attack_standard(model, test_dataloader, 0.5, 0.04, device, loss_fct=loss_fn)
adv_example_plot(examples, 'cifar_adv_test', transform=inv_scaling(), labels=test_data.classes)

logging.info('trials complete')
from torch.utils.data import DataLoader
from torchvision import datasets
from ..models import CifarCNN
from ..utils import *

set_up_log("cifar_clf_inf")

# set parameters
batch_size = 200
loss_fn = nn.CrossEntropyLoss()

logging.info("loading data")

# loading data
training_data = datasets.CIFAR10(
    root="../data",
    download=True,
    train=True,
    transform=eval_transform(),
)
test_data = datasets.CIFAR10(
    root="../data",
    download=True,
    train=False,
    transform=eval_transform(),
)

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info("loading data complete")
logging.info("preparing model")

# preparing device
device = get_device()

# preparing model
model = CifarCNN(n_channel=128).to(device)
model.load_state_dict(
    torch.load("../finals/CifarCNN.pth", map_location=device), strict=False
)

logging.info("model ready")
logging.info("running trials")

test_cl(train_dataloader, model, loss_fn, device, name="training")
test_cl(test_dataloader, model, loss_fn, device)

logging.info("trials complete")

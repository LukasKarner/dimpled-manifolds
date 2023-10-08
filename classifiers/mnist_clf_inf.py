from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ..models import MnistMLP
from utils import *
import logging

set_up_log("mnist_clf_inf")

# set parameters
batch_size = 200
loss_fn = nn.CrossEntropyLoss()

logging.info("loading data")

# loading data
training_data = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=eval_transform(1),
)
test_data = datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=eval_transform(1),
)

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info("loading data complete")
logging.info("preparing model")

# preparing device
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
logging.info(f"using {device} device")
device = torch.device(device)

# preparing model
model = MnistMLP().to(device)
model.load_state_dict(
    torch.load("finals/MnistMLP.pth", map_location=device), strict=True
)

logging.info("model ready")
logging.info("running trials")

test_cl(train_dataloader, model, loss_fn, device, name="training")
test_cl(test_dataloader, model, loss_fn, device)

logging.info("trials complete")

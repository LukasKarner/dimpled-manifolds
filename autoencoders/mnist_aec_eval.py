from torch.utils.data import DataLoader
from torchvision import datasets
from ..models import MnistAEC
from ..utils import *

log_to_stdout()

# set parameters
batch_size = 8

logging.info("loading data")

# loading data
training_data = datasets.MNIST(
    root="../data",
    download=True,
    train=True,
    transform=ToTensor(),
)
test_data = datasets.MNIST(
    root="../data",
    download=True,
    train=False,
    transform=ToTensor(),
)

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info("loading data complete")
logging.info("preparing model")

# preparing device
device = torch.device("cpu")

# preparing model
model = MnistAEC().to(device)
model.load_state_dict(
    torch.load("../finals/MnistAEC.pth", map_location=device), strict=True
)
model.eval()

logging.info("model ready")

for X, labels in train_dataloader:
    break
X = X.to(device)
Y = model(X)

X, Y = X.to("cpu").detach(), torch.clamp(Y.to("cpu").detach(), 0.0, 1.0)
aec_example_plot(X, Y, "train")

for X, labels in test_dataloader:
    break
X = X.to(device)
Y = model(X)

X, Y = X.to("cpu").detach(), torch.clamp(Y.to("cpu").detach(), 0.0, 1.0)
aec_example_plot(X, Y, "test")

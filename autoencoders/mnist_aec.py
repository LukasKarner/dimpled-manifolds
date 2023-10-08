from torch.utils.data import DataLoader
from torchvision import datasets
from ..models import MnistAEC
from ..utils import *

set_up_log("mnist_aec")

# set parameters
batch_size = 144
loss_fn = nn.MSELoss()
lr = 0.001
epochs = 1000

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
device = get_device()

# preparing model
model = MnistAEC().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

logging.info("model ready")
logging.info("optimising model")

# optimising model
for i in range(epochs):
    logging.info(f"epoch {i + 1}")
    train_ae(train_dataloader, model, loss_fn, optimizer, device)
    if (i + 1) % 10 == 0:
        test_ae(test_dataloader, model, loss_fn, device)

logging.info("optimisation complete")
logging.info("saving model")
torch.save(model.state_dict(), "MnistAEC.pth")
logging.info("model saved")

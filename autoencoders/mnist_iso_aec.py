from torch.utils.data import DataLoader
from torchvision import datasets
from ..models import MnistAEC
from utils import *

set_up_log("mnist_iso_aec")

# set parameters
batch_size = 16
loss_fn = nn.MSELoss()
lam = 100.0  # 1
lr = 0.01  # 0.001
epochs = 10

logging.info("loading data")

# loading data
training_data = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=ToTensor(),
)
test_data = datasets.MNIST(
    root="data",
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
test_iso_ae(
    train_dataloader, model, loss_fn, lam, device, name="train", verbose=11
)  # TODO remove
for i in range(epochs):
    logging.info(f"epoch {i + 1}")
    train_iso_ae(train_dataloader, model, loss_fn, lam, optimizer, device)
    if (i + 1) % 1 == 0:
        test_iso_ae(
            train_dataloader, model, loss_fn, lam, device, name="train", verbose=11
        )  # TODO remove

logging.info("optimisation complete")
logging.info("saving model")
torch.save(model.state_dict(), "MnistIAE.pth")
logging.info("model saved")

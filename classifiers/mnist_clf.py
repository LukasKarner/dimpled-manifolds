from torch.utils.data import DataLoader
from torchvision import datasets
from ..models import MnistMLP
from utils import *

set_up_log("mnist_clf")

# set parameters
batch_size = 200
loss_fn = nn.CrossEntropyLoss()
lr = 0.01
weight_decay = 0.0001
epochs = 40

logging.info("loading data")

# loading data
training_data = datasets.MNIST(
    root="data", download=True, train=True, transform=eval_transform(1)
)
test_data = datasets.MNIST(
    root="data", download=True, train=False, transform=eval_transform(1)
)

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info("loading data complete")
logging.info("preparing model")

# preparing device
device = get_device()

# preparing model
model = MnistMLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

logging.info("model ready")
logging.info("optimising model")

# optimising model
for i in range(epochs):
    logging.info(f"epoch {i+1}")
    train_cl(train_dataloader, model, loss_fn, optimizer, device)
    test_cl(test_dataloader, model, loss_fn, device)

logging.info("optimisation complete")
logging.info("saving model")
torch.save(model.state_dict(), "MnistMLP.pth")
logging.info("model saved")

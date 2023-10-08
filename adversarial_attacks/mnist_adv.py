from torch.utils.data import DataLoader
from torchvision import datasets
from ..models import MnistMLP, MnistAEC
from utils import *

set_up_log("mnist_adv")

# set parameters
batch_size = 1
loss_fn = MarginLoss()

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
train_dataloader = DataLoader(
    training_data,
    batch_size,
    True,
)
test_dataloader = DataLoader(
    test_data,
    batch_size,
    True,
)

logging.info("loading data complete")
logging.info("preparing model")

# preparing device
device = torch.device("cpu")

# preparing model
model = MnistMLP().to(device)
model.load_state_dict(
    torch.load("finals/MnistMLP.pth", map_location=device), strict=True
)

autoencoder = MnistAEC().to(device)
autoencoder.load_state_dict(
    torch.load("finals/MnistAEC.pth", map_location=device), strict=True
)

logging.info("model ready")
logging.info("running trials")

"""
examples = adv_attack_standard(model, train_dataloader, 2., 0.04, device, loss_fct=loss_fn)
adv_example_plot(examples, 'mnist_adv_train', transform=inv_scaling(1))

examples = adv_attack_standard(model, test_dataloader, 2., 0.04, device, loss_fct=loss_fn)
adv_example_plot(examples, 'mnist_adv_test', transform=inv_scaling(1))
"""

examples = adv_attack_manifold(
    model,
    autoencoder,
    transform=eval_transform_tensor(1),
    inv_transform=inv_scaling(1),
    dataloader=train_dataloader,
    epsilon=(2.0, 10.0, 10.0),
    step_size=(0.02, 0.2, 0.02),
    device=device,
)
adv_example_plot_projection(
    examples, "plots/temp/mnist/mnist_adv_train_", transform=inv_scaling(1)
)


examples = adv_attack_manifold(
    model,
    autoencoder,
    transform=eval_transform_tensor(1),
    inv_transform=inv_scaling(1),
    dataloader=test_dataloader,
    epsilon=(2.0, 10.0, 10.0),
    step_size=(0.02, 0.2, 0.02),
    device=device,
)
adv_example_plot_projection(
    examples, "plots/temp/mnist/mnist_adv_test_", transform=inv_scaling(1)
)

logging.info("trials complete")

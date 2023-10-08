from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from ..models import VGG16
from utils import *


set_up_log("imgnet_aec")
logging.getLogger("PIL.TiffImagePlugin").setLevel(51)

# set parameters
batch_size = 32
loss_fn = nn.MSELoss()
lr = 0.0001
epochs = 325

logging.info("loading data")

# loading data
training_data = datasets.ImageNet(
    root=get_imgnet_root(),
    split="train",
    transform=Compose(
        [models.ResNet50_Weights.DEFAULT.transforms(), inv_imgnet_scaling()]
    ),
)
# select only great white sharks and goldfish
training_classes = torch.tensor(training_data.targets)
training_ind = (training_classes == 1) | (training_classes == 2)
training_data = Subset(training_data, torch.nonzero(training_ind))

test_data = datasets.ImageNet(
    root=get_imgnet_root(),
    split="val",
    transform=Compose(
        [models.ResNet50_Weights.DEFAULT.transforms(), inv_imgnet_scaling()]
    ),
)
# select only great white sharks and goldfish
test_classes = torch.tensor(test_data.targets)
test_ind = (test_classes == 1) | (test_classes == 2)
test_data = Subset(test_data, torch.nonzero(test_ind))

# create dataloaders
train_dataloader = DataLoader(
    training_data,
    batch_size,
    True,
    pin_memory=True,
    num_workers=16,
    prefetch_factor=4,
)
test_dataloader = DataLoader(
    test_data, batch_size, True, pin_memory=True, num_workers=16, prefetch_factor=4
)

logging.info("loading data complete")
logging.info("preparing model")

# preparing device
device = get_device()

# preparing model
model = VGG16().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.8, patience=7, verbose=True
)

logging.info("model ready")
logging.info("optimising model")

# optimising model
for i in range(epochs):
    logging.info(f"epoch {i + 1}")
    train_ae(train_dataloader, model, loss_fn, optimizer, device)
    loss = test_ae(train_dataloader, model, loss_fn, device, "training")
    scheduler.step(loss)
    if (i + 1) % 10 == 0:
        test_ae(test_dataloader, model, loss_fn, device)

logging.info("optimisation complete")
logging.info("saving model")
torch.save(model.state_dict(), "ImgNetAEC.pth")
logging.info("model saved")

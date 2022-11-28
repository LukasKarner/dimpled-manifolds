from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from models import VGG16
from utils import *

log_to_stdout()
logging.getLogger("PIL.TiffImagePlugin").setLevel(51)

# set parameters
batch_size = 8

logging.info('loading data')

# loading data
training_data = datasets.ImageNet(
    root=get_imgnet_root(),
    split='train',
    transform=Compose([models.ResNet50_Weights.DEFAULT.transforms(), inv_imgnet_scaling()]),
)
training_classes = torch.tensor(training_data.targets)
training_ind = (training_classes == 1) | (training_classes == 2)
training_data = Subset(training_data, torch.nonzero(training_ind))

test_data = datasets.ImageNet(
    root=get_imgnet_root(),
    split='val',
    transform=Compose([models.ResNet50_Weights.DEFAULT.transforms(), inv_imgnet_scaling()]),
)
test_classes = torch.tensor(test_data.targets)
test_ind = (test_classes == 1) | (test_classes == 2)
test_data = Subset(test_data, torch.nonzero(test_ind))

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = get_device()

# preparing model
model = VGG16().to(device)
model.load_state_dict(torch.load('temp/ImgNetAEC.pth', map_location=device), strict=True)
model.eval()

logging.info('model ready')

for X, labels in train_dataloader:
    break
X = X.to(device)
Y = model(X)

X, Y = X.to('cpu').detach(), torch.clamp(Y.to('cpu').detach(), 0., 1.)
aec_example_plot(X, Y, 'train')

for X, labels in test_dataloader:
    break
X = X.to(device)
Y = model(X)

X, Y = X.to('cpu').detach(), torch.clamp(Y.to('cpu').detach(), 0., 1.)
aec_example_plot(X, Y, 'test')

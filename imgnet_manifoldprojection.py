from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from models import VGG16
from tqdm import trange
from utils import *

log_to_stdout()
logging.getLogger("PIL.TiffImagePlugin").setLevel(51)

# set parameters
batch_size = 1

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
model.load_state_dict(torch.load('finals/ImgNetAEC.pth', map_location=device), strict=True)
model.eval()

logging.info('model ready')

for X, labels in train_dataloader:
    break
X = X.to(device)
with torch.no_grad():
    Y = model.encoder(X)

Y.requires_grad_(True)
Z = model.decoder(Y)
n_pixels = Z.numel()
n_latents = Y.numel()
G = torch.empty(n_pixels, n_latents, device=device)
for i in trange(n_pixels):
    model.zero_grad()
    if i > 0:
        Y.grad *= 0.
    torch.flatten(Z)[i].backward(retain_graph=(i < n_pixels - 1))
    G[i] = torch.flatten(Y.grad)

H = G.T.resize(n_latents, 3, 224, 224)
assert torch.all(torch.flatten(Y.grad)[:n_latents] == H[:, 2, 223, 223]).item()
logging.info('beginning qr decomposition')
with torch.no_grad():
    R = in_place_qr(G)
    H = G.T.resize(n_latents, 3, 224, 224)
logging.info('qr decomposition complete')


for X, labels in test_dataloader:
    break
X = X.to(device)
with torch.no_grad():
    Y = model.encoder(X)

Y.requires_grad_(True)
Z = model.decoder(Y)
n_pixels = Z.numel()
n_latents = Y.numel()
G = torch.empty(n_pixels, n_latents, device=device)
for i in trange(n_pixels):
    model.zero_grad()
    if i > 0:
        Y.grad *= 0.
    torch.flatten(Z)[i].backward(retain_graph=(i < n_pixels - 1))
    G[i] = torch.flatten(Y.grad)

logging.info('beginning qr decomposition')
H = G.T.resize(n_latents, 3, 224, 224)
assert torch.all(torch.flatten(Y.grad)[:n_latents] == H[:, 2, 223, 223]).item()
with torch.no_grad():
    R = in_place_qr(G)
    H = G.T.resize(n_latents, 3, 224, 224)
logging.info('qr decomposition finished')

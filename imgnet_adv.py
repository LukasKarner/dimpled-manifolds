import torch
from torch.utils.data import DataLoader, Subset
from models import VGG16
from torchvision import datasets, models
from utils import *

set_up_log('imgnet_adv')
logging.getLogger("PIL.TiffImagePlugin").setLevel(51)

# set parameters
batch_size = 1
loss_fn = MarginLoss()

logging.info('loading data')

# loading data
training_data = datasets.ImageNet(
    root=get_imgnet_root(),
    split='train',
    transform=models.ResNet50_Weights.DEFAULT.transforms(),
)
train_labels = [i[0] for i in training_data.classes]
training_classes = torch.tensor(training_data.targets)
training_ind = (training_classes == 2)  # | (training_classes == 2)
training_data = Subset(training_data, torch.nonzero(training_ind))

test_data = datasets.ImageNet(
    root=get_imgnet_root(),
    split='val',
    transform=models.ResNet50_Weights.DEFAULT.transforms(),
)
test_labels = [i[0] for i in test_data.classes]
test_classes = torch.tensor(test_data.targets)
test_ind = (test_classes == 2)  # | (test_classes == 2)
test_data = Subset(test_data, torch.nonzero(test_ind))

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = get_device()

# preparing model
model = models.resnet50(models.ResNet50_Weights.DEFAULT, progress=False).to(device)

autoencoder = VGG16().to(device)
autoencoder.load_state_dict(torch.load('finals/ImgNetAEC.pth', map_location=device), strict=True)

logging.info('model ready')
logging.info('running trials')

'''
examples = adv_attack_standard(model, train_dataloader, 2., 0.8, device, loss_fct=loss_fn)
adv_example_plot(examples, 'imgnet_adv_train', transform=inv_imgnet_scaling(), labels=[i[0] for i in training_data.classes])

examples = adv_attack_standard(model, test_dataloader, 2., 0.8, device, loss_fct=loss_fn)
adv_example_plot(examples, 'imgnet_adv_test', transform=inv_imgnet_scaling(), labels=[i[0] for i in test_data.classes])
'''

examples = adv_attack_manifold(
    model,
    autoencoder,
    transform=imgnet_scaling(),
    inv_transform=inv_imgnet_scaling(),
    dataloader=train_dataloader,
    epsilon=(2., 5., 5.),
    step_size=(0.01, 0.1, 0.01),
    device=device,
    max_n=1,
    target=2,
)
adv_example_plot_projection(
    examples,
    'plots/temp/imgnet/imgnet_adv_train_',
    transform=inv_imgnet_scaling(),
    labels=train_labels,
)


examples = adv_attack_manifold(
    model,
    autoencoder,
    transform=imgnet_scaling(),
    inv_transform=inv_imgnet_scaling(),
    dataloader=test_dataloader,
    epsilon=(2., 5., 5.),
    step_size=(0.01, 0.1, 0.01),
    device=device,
    max_n=1,
    target=2,
)
adv_example_plot_projection(
    examples,
    'plots/temp/imgnet/imgnet_adv_test_',
    transform=inv_imgnet_scaling(),
    labels=test_labels,
)

logging.info('trials complete')

from torch.utils.data import DataLoader
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
test_data = datasets.ImageNet(
    root=get_imgnet_root(),
    split='val',
    transform=models.ResNet50_Weights.DEFAULT.transforms(),
)

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = get_device()

# preparing model
model = models.resnet50(models.ResNet50_Weights.DEFAULT, progress=False).to(device)

logging.info('model ready')
logging.info('running trials')

examples = adv_attack_standard(model, train_dataloader, 2., 0.8, device, loss_fct=loss_fn)
adv_example_plot(examples, 'imgnet_adv_train', transform=inv_imgnet_scaling(), labels=[i[0] for i in training_data.classes])

examples = adv_attack_standard(model, test_dataloader, 2., 0.8, device, loss_fct=loss_fn)
adv_example_plot(examples, 'imgnet_adv_test', transform=inv_imgnet_scaling(), labels=[i[0] for i in test_data.classes])

logging.info('trials complete')

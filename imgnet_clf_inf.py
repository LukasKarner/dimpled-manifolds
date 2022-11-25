from torch.utils.data import DataLoader
from torchvision import datasets, models
from utils import *

if __name__ == '__main__':

    set_up_log('imgnet_clf_inf')

    # set parameters
    batch_size = 32
    loss_fn = nn.CrossEntropyLoss()

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
    train_dataloader = DataLoader(training_data, batch_size, True, pin_memory=True, num_workers=8, prefetch_factor=4)
    test_dataloader = DataLoader(test_data, batch_size, True, pin_memory=True, num_workers=8, prefetch_factor=4)

    logging.info('loading data complete')
    logging.info('preparing model')

    # preparing device
    device = get_device()

    # preparing model
    model = models.resnet50(models.ResNet50_Weights.DEFAULT, progress=False).to(device)

    logging.info('model ready')
    logging.info('running trials')

    test_cl(test_dataloader, model, loss_fn, device, verbose=6)
    test_cl(train_dataloader, model, loss_fn, device, name='training', verbose=6)

    logging.info('trials complete')

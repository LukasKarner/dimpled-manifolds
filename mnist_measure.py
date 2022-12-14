from torch.utils.data import DataLoader
from torchvision import datasets
from models import MnistMLP
from utils import *
import numpy as np

set_up_log('mnist_measure')

# set parameters
batch_size = 200
attack_size = 1
loss_fn = nn.CrossEntropyLoss()
loss_fn_a = MarginLoss()  # TODO
lr = 0.01
weight_decay = 0.0001
epochs = 40  # TODO

logging.info('loading data')

# loading data
training_data = datasets.MNIST(
    root='data.nosync',
    download=True,
    train=True,
    transform=eval_transform(1)
)
test_data = datasets.MNIST(
    root='data.nosync',
    download=True,
    train=False,
    transform=eval_transform(1)
)

# create dataloaders
train_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, attack_size, True)

logging.info('loading data complete')
logging.info('preparing model')

# preparing device
device = get_device()

# preparing model
model = MnistMLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

logging.info('model ready')
logging.info('optimising model')

distance_list = []

# optimising model
for i in range(epochs):
    logging.info(f'epoch {i+1}')
    train_cl(train_dataloader, model, loss_fn, optimizer, device)
    examples = adv_attack_standard(
        model,
        test_dataloader,
        epsilon=5.,
        step_size=0.04,
        max_iter=1000,
        device=device,
        loss_fct=loss_fn_a,
        measurement=False,  # TODO
        max_n=200,
    )
    d = np.array([torch.norm(t[0] - t[1]).item() for t in examples])
    distance_list.append([np.mean(d), np.std(d)])

distance_list = np.array(distance_list)
fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(x=np.arange(epochs), y1=distance_list[:,0] + distance_list[:,1], y2=distance_list[:,0] - distance_list[:,1], step='mid', alpha=0.5)
ax.plot(distance_list[:, 0])
ax.set_xlabel('Epochs')
ax.set_title('Distance of adversarial examples to inputs during training of a MNIST classifier')
ax.set_ylabel('l_2 distance')
plt.tight_layout()
plt.savefig('mnist_measure.pdf')

logging.info('optimisation complete')

logging.info('trials complete')

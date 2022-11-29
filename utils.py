import torch
from torch import nn
from torchvision.transforms import ToTensor, Pad, RandomCrop, RandomHorizontalFlip, Normalize, Compose
import matplotlib.pyplot as plt
import logging
import sys
import platform
import os


#####################
# general utilities #
#####################


def get_device():
    device = 'mps' if torch.backends.mps.is_available() \
        else 'cuda' if torch.cuda.is_available() \
        else 'cpu'
    if device == 'cuda':
        n = torch.cuda.device_count()
        if n > 1:
            i = input(f'enter cuda index ({n} devices available): ')
            assert isinstance(int(i), int)
            device = device + ':' + i
    logging.info(f'using {device} device')
    return torch.device(device)


def get_imgnet_root():
    system = platform.system()
    if system == 'Darwin':
        return 'data.nosync'
    elif system == 'Linux':
        return f'/media/data/{os.getlogin()}_data'
    else:
        raise Exception


#########################
# image transformations #
#########################


def train_transform(size):
    return Compose([Pad(4),
                    RandomCrop(size),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                   )


def eval_transform(n_channels: int = 3):
    return Compose([ToTensor(),
                    Normalize([0.5] * n_channels, [0.5] * n_channels)]
                   )


def inv_scaling(n_channels: int = 3):
    return Normalize([-1] * n_channels, [2] * n_channels)


def inv_imgnet_scaling():
    return Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))


#########################
# functions for logging #
#########################


def set_up_log(filename):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.DEBUG,
        filename=filename + '.log',
    )
    plt.set_loglevel('warning')


def log_to_stdout():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    plt.set_loglevel('warning')


#############################################
# functions for training and testing models #
#############################################


def train_ae(
        dataloader,
        model,
        loss_fn,
        optimizer,
        device,
        verbose: int = 6
):
    size = len(dataloader)
    if verbose:
        checkpoints = torch.linspace(0, size - 1, verbose, dtype=int)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward
        pred = model(X)
        loss = loss_fn(pred, X)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and batch in checkpoints:
            loss, pct = loss.item(), batch / size
            logging.info(f'{pct:4.0%} | loss = {loss:f}')


def test_ae(
        dataloader,
        model,
        loss_fn,
        device,
        name='test',
        verbose: int = None,
):
    num_batches = len(dataloader)
    if verbose:
        checkpoints = torch.linspace(0, num_batches - 1, verbose, dtype=int)
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, X).item()
            if verbose and batch in checkpoints:
                logging.info(f'{name} progress: {batch/num_batches:.0%}')
    loss /= num_batches
    logging.info(f'{name} results: loss = {loss:f}')
    return loss


def train_cl(
        dataloader,
        model,
        loss_fn,
        optimizer,
        device,
        verbose: int = 6,
):
    size = len(dataloader)
    if verbose:
        checkpoints = torch.linspace(0, size - 1, verbose, dtype=int)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and batch in checkpoints:
            loss, pct, acc = loss.item(), batch / size, (pred.argmax(1) == y).type(torch.float).mean().item()
            logging.info(f'{pct:4.0%} | accuracy = {acc:7.2%} | loss = {loss:f}')


def test_cl(
        dataloader,
        model,
        loss_fn,
        device,
        name='test',
        verbose: int = None,
):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if verbose:
        checkpoints = torch.linspace(0, num_batches - 1, verbose, dtype=int)
    model.eval()
    loss = correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if verbose and batch in checkpoints:
                logging.info(f'{name} progress: {batch/num_batches:.0%}')
    loss /= num_batches
    correct /= size
    logging.info(f'{name} results: accuracy = {correct:%}, loss = {loss:f}')
    return correct, loss


####################################
# functions for visualising images #
####################################


def get_pred(logits):
    sm = nn.Softmax(1)
    probs = sm(logits).squeeze()
    return probs.argmax().item(), probs.max().item(), probs


def adv_example_plot(examples, name=None, transform=None, labels=None):
    n = len(examples)
    fig, axs = plt.subplots(n, 3, squeeze=False, figsize=(9, n*3))
    for i in range(n):
        x, perturbed, logits_o, logits_a = examples[i]
        if transform:
            x, perturbed = transform(x), transform(perturbed)
        if len(x.size()) == 4:
            x, perturbed = torch.permute(x, (0, 2, 3, 1)), torch.permute(perturbed, (0, 2, 3, 1))
        delta = perturbed - x
        d = torch.norm(delta)
        delta = (delta - delta.min()) / (delta.max() - delta.min())
        ax = axs[i]
        ax[0].imshow(x.squeeze(), cmap='gray', vmin=0., vmax=1.)
        y, p, _ = get_pred(logits_o)
        if labels:
            y = labels[y]
        ax[0].set_title(f'label {y} w.p. {p:.3f}')
        ax[0].axis('off')
        ax[1].imshow(perturbed.squeeze(), cmap='gray', vmin=0., vmax=1.)
        y, p, _ = get_pred(logits_a)
        if labels:
            y = labels[y]
        ax[1].set_title(f'label {y} w.p. {p:.3f}')
        ax[1].axis('off')
        ax[2].imshow(delta.squeeze(), cmap='gray', vmin=0., vmax=1.)
        ax[2].set_title(f'difference l_2 norm = {d:.3f}')
        ax[2].axis('off')
    plt.tight_layout()
    if name:
        plt.savefig(name+'.pdf')
    else:
        plt.show()


def aec_example_plot(X: torch.Tensor, Y: torch.Tensor, name: str = None, transform=None):
    n = len(X)
    assert n == len(Y)
    if transform:
        X, Y = transform(X), transform(Y)
    if len(X.size()) == 4:
        X, Y = torch.permute(X, (0, 2, 3, 1)), torch.permute(Y, (0, 2, 3, 1))
    fig, axs = plt.subplots(n, 2, squeeze=False, figsize=(9, n * 4.5))
    for i in range(n):
        x, y = X[i], Y[i]
        ax = axs[i]
        d_2 = torch.norm(x-y)
        d_s = torch.norm(x-y, float('inf'))
        ax[0].imshow(x.squeeze(), cmap='gray', vmin=0., vmax=1.)
        ax[0].set_title(f'original\nl_2 distance = {d_2:.5f}')
        ax[0].axis('off')
        ax[1].imshow(y.squeeze(), cmap='gray', vmin=0., vmax=1.)
        ax[1].set_title(f'reconstruction\nl_inf distance = {d_s:.5f}')
        ax[1].axis('off')
    plt.tight_layout()
    if name:
        plt.savefig(name + '.pdf')
    else:
        plt.show()


###########################################
# functions regarding adversarial attacks #
###########################################


class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(x - x[0, y.item()])  # TODO multi dim


def pgd_attack(
        x: torch.Tensor,
        model,
        epsilon,
        step_size,
        loss_fn=nn.CrossEntropyLoss(),
        target=None,
        manifold_projection=None,
        max_iter=50,
):
    x_ = x.clone().detach()
    device = x_.device
    assert epsilon > 0

    def projection(t):
        d = torch.norm(t - x)
        if d <= epsilon:
            return t
        else:
            return epsilon * t / d + x * (1 - epsilon / d)

    if x.size()[-1] == 224:  # detect ImageNet input
        mins = torch.tensor([-2.11790393, -2.03571429, -1.80444444]).resize_(3, 1, 1).to(device)
        mins = torch.ones_like(x_) * mins
        maxs = torch.tensor([2.2489083, 2.42857143, 2.64]).resize_(3, 1, 1).to(device)
        maxs = torch.ones_like(x_) * maxs
    else:
        mins = 1.
        maxs = 1.

    targeted = bool(target)
    model.eval()
    with torch.no_grad():
        if not targeted:
            target = model(x).argmax(1)
    assert step_size > 0
    step_size = step_size if targeted else -1 * step_size

    for i in range(max_iter):
        x_.requires_grad_(True)
        pred = model(x_)
        if targeted ^ (pred.argmax(1).item() != target.item()):
            logging.info(f'pgd attack successful after {i} iterations')
            return x_.clone().detach()
        loss = loss_fn(pred, target)
        model.zero_grad()
        loss.backward()
        grad = x_.grad.data
        with torch.no_grad():
            grad = manifold_projection(grad) if manifold_projection else grad
            grad_norm = torch.norm(grad)
            if grad_norm.item() == 0:
                logging.info('pgd attack not successful: gradient == 0')
                return x_.clone().detach()
            grad = grad / grad_norm
            x_ = x_ - step_size * grad
            _ = torch.norm(x - x_)
            x_ = torch.clamp(projection(x_), mins, maxs)
    logging.info('pgd attack reached max_iter')
    return x_.clone().detach()


def adv_attack_standard(
        model,
        dataloader,
        epsilon,
        step_size,
        device,
        max_n=10,
        max_iter=50,
        loss_fct=nn.CrossEntropyLoss()
):
    assert epsilon > 0
    model.eval()
    examples = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred_o = model(x)
            if pred_o.argmax(1).item() != y.item():
                continue
        perturbed = pgd_attack(x, model, epsilon, step_size, max_iter=max_iter, loss_fn=loss_fct)
        with torch.no_grad():
            pred_a = model(perturbed)
            if pred_a.argmax(1).item() != y.item():
                examples.append((x.cpu().detach(), perturbed.cpu().detach(), pred_o.cpu(), pred_a.cpu()))
                logging.info('adversarial example found')
                if len(examples) == max_n:
                    break
    return examples

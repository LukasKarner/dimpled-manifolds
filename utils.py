import torch
from torch import nn
from numpy import linspace
import matplotlib.pyplot as plt
import logging
import sys


def get_device():
    device = 'mps' if torch.backends.mps.is_available() \
        else 'cuda' if torch.cuda.is_available() \
        else 'cpu'
    if device == 'cuda':
        n = torch.cuda.device_count()
        if n > 0:
            i = input(f'enter cuda index ({n} devices available): ')
            assert isinstance(int(i), int)
            device = device + ':' + i
    logging.info(f'using {device} device')
    return torch.device(device)


#########################
# functions for logging #
#########################


def set_up_log(filename):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.DEBUG,
        filename=filename + '.log',
        encoding='utf-8',
    )
    plt.set_loglevel('warning')


def log_to_stdout():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.DEBUG,
        stream=sys.stdout,
        encoding='utf-8',
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
):
    size = len(dataloader)
    checkpoints = linspace(0, size - 1, 6, dtype=int)
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

        if batch in checkpoints:
            loss, pct = loss.item(), batch / size
            logging.info(f'{pct:4.0%} | loss = {loss:f}')


def test_ae(
        dataloader,
        model,
        loss_fn,
        device,
        name='test',
):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, X).item()
    loss /= num_batches
    logging.info(f'{name} results: loss = {loss:f}')
    return loss


def train_cl(
        dataloader,
        model,
        loss_fn,
        optimizer,
        device,
):
    size = len(dataloader)
    checkpoints = linspace(0, size - 1, 6, dtype=int)
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

        if batch in checkpoints:
            loss, pct, acc = loss.item(), batch / size, (pred.argmax(1) == y).type(torch.float).mean().item()
            logging.info(f'{pct:4.0%} | accuracy = {acc:7.2%} | loss = {loss:f}')


def test_cl(
        dataloader,
        model,
        loss_fn,
        device,
        name='test',
):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss = correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    logging.info(f'{name} results: accuracy = {correct:%}, loss = {loss:f}')
    return correct, loss


####################################
# functions for visualising images #
####################################


def get_pred(logits):
    sm = nn.Softmax()
    probs = sm(logits)
    return probs.argmax(), probs.max()


def adv_example_plot(examples, name=None):
    n = len(examples)
    fig, axs = plt.subplots(n, 3, squeeze=False, figsize=(9, n*3))
    for i in range(n):
        x, perturbed, logits_o, logits_a = examples[i]
        delta = perturbed - x
        # TODO add confidences
        ax = axs[i]
        ax[0].imshow(x.squeeze(), cmap='gray', vmin=0., vmax=1.)
        y, p = get_pred(logits_o)
        ax[0].set_title(f'{y} w.p. {p:.3f}')
        ax[0].axis('off')
        ax[1].imshow(perturbed.squeeze(), cmap='gray', vmin=0., vmax=1.)
        y, p = get_pred(logits_a)
        ax[1].set_title(f'{y} w.p. {p:.3f}')
        ax[1].axis('off')
        ax[2].imshow(delta.squeeze(), cmap='gray')
        ax[2].set_title(f'l_2 norm = {torch.norm(delta):.3f}')
        ax[2].axis('off')
    plt.tight_layout()
    if name:
        plt.savefig(name+'.pdf')
    plt.show()


###########################################
# functions regarding adversarial attacks #
###########################################


def pgd_attack(
        x,
        model,
        epsilon,
        loss_fn=nn.CrossEntropyLoss(),
        target=None,
        norm='2',
        manifold_projection=None,
        max_iter=50,
):
    x_orig = torch.clone(x).detach()
    assert norm == '2', 'inf-norm not yet implemented'

    def projection(t):
        d = torch.norm(t - x_orig, p=float(norm))
        if d <= epsilon:
            return t
        else:
            # TODO implement inf-norm-projection
            return epsilon * t / d + x_orig * (1 - epsilon / d)

    targeted = bool(target)
    model.eval()
    if not targeted:
        target = model(x).argmax(1)
    assert epsilon > 0
    epsilon = epsilon if targeted else -1 * epsilon

    for i in range(max_iter):
        x.requires_grad = True
        pred = model(x)
        if targeted ^ (pred.argmax(1).item() != target.item()):
            logging.info(f'pgd attack successful after {i} iterations')
            return x.detach()
        loss = loss_fn(pred, target)
        model.zero_grad()
        loss.backward()
        grad = x.grad.data
        grad = manifold_projection(grad) if manifold_projection else grad
        grad_norm = torch.norm(grad, p=float(norm))
        if grad_norm.item() == 0:
            logging.info('pgd attack not successful: gradient = 0')
            return x.detach()
        grad = grad / grad_norm  # TODO implement inf-norm normalisation
        x = x - epsilon * grad
        x = torch.clamp(projection(x), 0, 1).detach()
    logging.info('pgd attack reached max_iter')
    return x.detach()


def adv_attack_standard(
        model,
        dataloader,
        epsilon,
        device,
        max_n=10,
        max_iter=50,
):
    assert epsilon > 0
    model.eval()
    examples = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred_o = model(x)
        if pred_o.argmax(1).item() != y.item():
            continue
        perturbed = pgd_attack(x, model, epsilon, max_iter=max_iter)
        pred_a = model(perturbed)
        if pred_a.argmax(1).item() != y.item():
            examples.append((x.cpu().detach(), perturbed.cpu().detach(), pred_o.cpu(), pred_a.cpu()))
            logging.info('adversarial example found')
            if len(examples) == max_n:
                break
    return examples

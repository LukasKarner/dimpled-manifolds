import torch
from numpy import linspace
import logging


def set_up_log(filename):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.DEBUG,
        filename=filename+'.log',
        encoding='utf-8',
    )


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
    logging.info(f'test results: loss = {loss:f}')


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
            logging.info(f'{pct:4.0%} | accuracy = {acc:4.2%} | loss = {loss:f}')


def test_cl(
        dataloader,
        model,
        loss_fn,
        device,
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
    logging.info(f'test results: accuracy = {correct:4.2%}, loss = {loss:f}')

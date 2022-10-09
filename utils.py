import torch
from numpy import linspace


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader)
    checkpoints = linspace(0, size - 1, 11, dtype=int)
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
            loss, pct = loss.item(), batch/size
            print(f'{pct:.0%} | loss = {loss:f}')


def test_cl(dataloader, model, loss_fn, device):
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
    print(f'test results: accuracy = {correct:.2%}, loss = {loss:f}')

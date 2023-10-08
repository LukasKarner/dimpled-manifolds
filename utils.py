import torch
from torch import nn
from torchvision.transforms import (
    ToTensor,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    Normalize,
    Compose,
)
import functorch as ft
import matplotlib.pyplot as plt
import logging
import sys
import platform
import os
from models import IsoLoss, PIsoLoss


####################################################################################################
# general utilities                                                                                #
####################################################################################################


def get_device() -> torch.device:
    """Returns a PyTorch device object based on the availability of MPS, CUDA, or CPU.
    If CUDA is available and there are multiple devices, the user is prompted to select a device index.
    """
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device == "cuda":
        n = torch.cuda.device_count()
        if n > 1:
            i = input(f"enter cuda index ({n} devices available): ")
            assert isinstance(int(i), int)
            device = device + ":" + i

    logging.info(f"using {device} device")

    return torch.device(device)


def get_imgnet_root() -> str:
    """Returns the root directory for the ImageNet dataset based on the current operating system.

    Returns:
        str: The root directory for the ImageNet dataset.

    Raises:
        Exception: If the current operating system is not supported.
    """
    system = platform.system()

    if system == "Darwin":
        return "../data"
    elif system == "Linux":
        return f"/media/data/{os.getlogin()}_data"
    else:
        raise Exception


####################################################################################################
# image transformations                                                                            #
####################################################################################################


def train_transform(size: int) -> Compose:
    """Returns a composition of image transformations for training data.

    Args:
        size (int): The size of the output image after cropping.

    Returns:
        Compose: A composition of image transformations.
    """
    return Compose(
        [
            Pad(4),
            RandomCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def eval_transform(n_channels: int = 3) -> Compose:
    """Returns a torchvision.transforms.Compose object for inference time transformations.

    Args:
        n_channels (int): The number of channels in the input image. Default is 3.

    Returns:
        A torchvision.transforms.Compose object.
    """
    return Compose([ToTensor(), Normalize([0.5] * n_channels, [0.5] * n_channels)])


def eval_transform_tensor(n_channels: int = 3) -> Normalize:
    """Returns a PyTorch transform that normalizes a tensor of shape (C, H, W) with mean 0.5 and standard deviation 0.5
    for each channel.

    Args:
        n_channels (int): Number of channels in the input tensor. Default is 3 for RGB images.

    Returns:
        torchvision.transforms.Normalize: PyTorch transform that normalizes a tensor of shape (C, H, W) with mean 0.5
        and standard deviation 0.5 for each channel.
    """
    return Normalize([0.5] * n_channels, [0.5] * n_channels)


def inv_scaling(n_channels: int = 3) -> Normalize:
    """
    Returns a normalization function that scales pixel values from [-1, 1] to [0, 1].

    Args:
        n_channels (int): The number of color channels in the image. Default is 3 for RGB images.

    Returns:
        A normalization function that scales pixel values from [-1, 1] to [0, 255].
    """
    return Normalize([-1] * n_channels, [2] * n_channels)


def imgnet_scaling() -> Normalize:
    """Returns a torchvision Normalize transform that scales image tensors to have a mean of
    (0.485, 0.456, 0.406) and a standard deviation of (0.229, 0.224, 0.225).
    """
    return Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def inv_imgnet_scaling() -> Normalize:
    """Returns a normalization function that undoes the scaling applied to images in the
    ImageNet dataset.
    """
    return Normalize(
        (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        (1 / 0.229, 1 / 0.224, 1 / 0.225),
    )


####################################################################################################
# functions for logging                                                                           #
####################################################################################################


def set_up_log(filename: str) -> None:
    """Sets up logging to filename.log ."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG,
        filename=filename + ".txt",
    )
    plt.set_loglevel("warning")


def log_to_stdout() -> None:
    """Sets up logging to stdout."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    plt.set_loglevel("warning")


####################################################################################################
# functions for training and testing models                                                        #
####################################################################################################


def train_ae(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    verbose: int = 6,
) -> None:
    """
    Trains an autoencoder model on a given dataloader using a specified loss function and optimizer.

    Args:
        dataloader (DataLoader): The dataloader to use for training.
        model (Module): The autoencoder model to train.
        loss_fn: The loss function to use for training.
        optimizer (Optimizer): The optimizer to use for training.
        device (str): The device to use for training (e.g. 'cpu', 'cuda').
        verbose (int, optional): The number of verbose steps to print during training. Defaults to 6.
    """
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
            logging.info(f"{pct:4.0%} | loss = {loss:f}")


def train_iso_ae(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    lam: float,
    optimizer: torch.optim.Optimizer,
    device: str,
    verbose: int = 6,
) -> None:
    """
    Trains an Isometric Autoencoder (IsoAE) on the given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader to train on.
        model (torch.nn.Module): The IsoAE model to train.
        loss_fn (torch.nn.Module): The loss function to use.
        lam (float): The lambda value for the IsoLoss and PIsoLoss functions.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (str): The device to use for training.
        verbose (int, optional): The verbosity level. Defaults to 6.
    """
    iso_loss = IsoLoss(lam, device).to(device)
    piso_loss = PIsoLoss(lam, device).to(device)

    size = len(dataloader)
    if verbose:
        checkpoints = torch.linspace(0, size - 1, verbose, dtype=int)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward
        model.train()
        lat = model.encoder(X)
        pred = model.decoder(lat)
        loss = loss_fn(pred, X)

        # backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # prepare tensors for jacobian computation
        X = torch.unsqueeze(X, 1)
        lat = torch.unsqueeze(lat, 1)

        model.eval()  # necessary to make batchnorm work with jacobian computation
        encoder_jac = ft.vmap(ft.jacrev(model.encoder))(X)
        decoder_jac = ft.vmap(ft.jacfwd(model.decoder))(lat)

        # forward again
        l_iso = iso_loss(decoder_jac)
        l_piso = piso_loss(encoder_jac)

        # backward again
        l_iso.backward(retain_graph=True)
        l_piso.backward()

        optimizer.step()

        if verbose and batch in checkpoints:
            loss, i_loss, p_loss, pct = (
                loss.item(),
                l_iso.item(),
                l_piso.item(),
                batch / size,
            )
            logging.info(
                f"{pct:4.0%} | mse = {loss:f} | iso = {i_loss:f} | piso = {p_loss:f}"
            )


def test_ae(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    name: str = "test",
    verbose: int = None,
) -> float:
    """Test an autoencoder model on a given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader to use for testing.
        model (torch.nn.Module): The autoencoder model to test.
        loss_fn (callable): The loss function to use for testing.
        device (torch.device): The device to use for testing.
        name (str, optional): The name of the test. Defaults to "test".
        verbose (int, optional): The number of progress checkpoints to log during testing. Defaults to None.

    Returns:
        float: The average loss over the test set.
    """
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
                logging.info(f"{name} progress: {batch / num_batches:.0%}")

    loss /= num_batches
    logging.info(f"{name} results: loss = {loss:f}")
    return loss


def test_iso_ae(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    lam: float,
    device: torch.device,
    name: str = "test",
    verbose: int = None,
) -> tuple[float, float, float]:
    """Test an isometric autoencoder.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for the test data.
        model (torch.nn.Module): The autoencoder model to test.
        loss_fn (torch.nn.Module): The loss function to use for the reconstruction loss.
        lam (float): The lambda value to use for the isotropy and partial isotropy loss functions.
        device (torch.device): The device to use for the computations.
        name (str, optional): The name of the test. Defaults to "test".
        verbose (int, optional): The number of checkpoints to log progress at. Defaults to None.

    Returns:
        Tuple[float, float, float]: A tuple containing the average reconstruction loss, isometry loss, and inverse isometry loss.
    """
    iso_loss = IsoLoss(lam, device).to(device)
    piso_loss = PIsoLoss(lam, device).to(device)

    num_batches = len(dataloader)
    if verbose:
        checkpoints = torch.linspace(0, num_batches - 1, verbose, dtype=int)

    model.eval()
    loss = 0
    l_iso = 0
    l_piso = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward
        with torch.no_grad():
            lat = model.encoder(X)
            pred = model.decoder(lat)
            loss += loss_fn(pred, X).item()

        # prepare tensors for jacobian computation
        X = torch.unsqueeze(X, 1)
        lat = torch.unsqueeze(lat, 1)

        # backward
        encoder_jac = ft.vmap(ft.jacrev(model.encoder))(X)
        decoder_jac = ft.vmap(ft.jacfwd(model.decoder))(lat)

        # forward again
        l_iso += iso_loss(decoder_jac).item()
        l_piso += piso_loss(encoder_jac).item()

        if verbose and batch in checkpoints:
            logging.info(f"{name} progress: {batch / num_batches:.0%}")

    loss /= num_batches
    l_iso /= num_batches
    l_piso /= num_batches

    logging.info(
        f"{name} results: mse = {loss:f} | iso = {l_iso:f} | piso = {l_piso:f}"
    )
    return loss, l_iso, l_piso


def train_cl(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    verbose: int = 6,
) -> None:
    """Trains a classification model on a given dataset using mini-batch stochastic gradient descent.

    Args:
        dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object that provides the training data.
        model (nn.Module): A PyTorch module that represents the classification model to be trained.
        loss_fn (nn.Module): A PyTorch module that computes the loss function for the classification task.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer object that updates the model parameters during training.
        device (str): A string that specifies the device (e.g., 'cpu' or 'cuda') on which to run the training.
        verbose (int, optional): An integer that specifies the number of progress checkpoints to print during training.
            Defaults to 6.
    """
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
            loss, pct, acc = (
                loss.item(),
                batch / size,
                (pred.argmax(1) == y).type(torch.float).mean().item(),
            )
            logging.info(f"{pct:4.0%} | accuracy = {acc:7.2%} | loss = {loss:f}")


def test_cl(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    name: str = "test",
    verbose: int = None,
) -> tuple[float, float]:
    """Test the accuracy and loss of a model on a given dataset.

    Args:
        dataloader (DataLoader): The data loader for the dataset.
        model (torch.nn.Module): The model to be tested.
        loss_fn (torch.nn.Module): The loss function to be used.
        device (torch.device): The device to run the model on.
        name (str, optional): The name of the test. Defaults to "test".
        verbose (int, optional): The number of progress checkpoints to log. Defaults to None.

    Returns:
        Tuple[float, float]: The accuracy and loss of the model on the dataset.
    """
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
                logging.info(f"{name} progress: {batch / num_batches:.0%}")

    loss /= num_batches
    correct /= size

    logging.info(f"{name} results: accuracy = {correct:%}, loss = {loss:f}")
    return correct, loss


####################################################################################################
# functions for visualising images                                                                 #
####################################################################################################


def adv_example_plot(
    examples: list[tuple[torch.Tensor]],
    name: str = None,
    transform: nn.Module = None,
    labels: list[str] = None,
) -> None:
    """
    Visualise a list of adversarial attacks on images.

    Args:
        examples (list[tuple[torch.Tensor]]): A list of tuples containing the original image, perturbed image, logits of the original image, and logits of the perturbed image.
        name (str): The name of the file to save the plot as a PDF. If None, the plot will be displayed.
        transform (nn.Module): A PyTorch module to transform the images before plotting.
        labels (list[str]): A list of labels for the classes in the logits. If None, the class index will be used as the label.
    """
    n = len(examples)
    fig, axs = plt.subplots(n, 3, squeeze=False, figsize=(9, n * 3))
    for i in range(n):
        x, perturbed, logits_o, logits_a = examples[i]
        if transform:
            x, perturbed = transform(x), transform(perturbed)
        if len(x.size()) == 4:
            x, perturbed = torch.permute(x, (0, 2, 3, 1)), torch.permute(
                perturbed, (0, 2, 3, 1)
            )
        delta = perturbed - x
        d = torch.norm(delta)
        delta = (delta - delta.min()) / (delta.max() - delta.min())
        ax = axs[i]
        ax[0].imshow(x.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        y, p, _ = get_pred(logits_o)
        if labels:
            y = labels[y]
        ax[0].set_title(f"label {y} w.p. {p:.3f}")
        ax[0].axis("off")
        ax[1].imshow(perturbed.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        y, p, _ = get_pred(logits_a)
        if labels:
            y = labels[y]
        ax[1].set_title(f"label {y} w.p. {p:.3f}")
        ax[1].axis("off")
        ax[2].imshow(delta.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        ax[2].set_title(f"difference l_2 norm = {d:.3f}")
        ax[2].axis("off")
    plt.tight_layout()
    if name:
        plt.savefig(name + ".pdf")
    else:
        plt.show()


def adv_example_plot_projection(
    examples=list[tuple[torch.Tensor]],
    name: str = None,
    transform: nn.Module = None,
    labels: list[str] = None,
) -> None:
    """Visualise a list of adversarial attacks on images and their projections onto/off the manifold.

    Args:
        examples (list[tuple[torch.Tensor]]): A list of adversarial attacks on images.
        name (str): A string to be used as a prefix for the names of the generated plots. If None, the plot will be displayed.
        transform (nn.Module): A PyTorch module to be used for transforming the images before plotting.
        labels (list[str]): A list of strings to be used as labels for the generated plot.
    """
    for i in range(len(examples)):
        c = examples[i]
        one_example = [(c[0], c[j], c[4], c[4 + j]) for j in range(1, 4)]
        adv_example_plot(
            one_example, name=name + str(i), transform=transform, labels=labels
        )


def aec_example_plot(
    X: torch.Tensor, Y: torch.Tensor, name: str = None, transform: nn.Module = None
) -> None:
    """
    Visualize a list of original and reconstructed images from an autoencoder.

    Args:
    - X (torch.Tensor): A tensor of shape (n, c, h, w) containing the original images.
    - Y (torch.Tensor): A tensor of shape (n, c, h, w) containing the reconstructed images.
    - name (str): Optional. The name of the file to save the plot as a PDF. If None, the plot will be displayed.
    - transform (nn.Module): Optional. A PyTorch module to apply to the images before plotting.

    Raises:
    - AssertionError: If the length of X and Y are not equal.
    """
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
        d_2 = torch.norm(x - y)
        d_s = torch.norm(x - y, float("inf"))
        ax[0].imshow(x.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        ax[0].set_title(f"original\nl_2 distance = {d_2:.5f}")
        ax[0].axis("off")
        ax[1].imshow(y.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        ax[1].set_title(f"reconstruction\nl_inf distance = {d_s:.5f}")
        ax[1].axis("off")
    plt.tight_layout()
    if name:
        plt.savefig(name + ".pdf")
    else:
        plt.show()


def get_pred(logits: torch.Tensor) -> torch.Tensor:
    """Get probabilities from logits.

    Args:
        logits (torch.Tensor): The logits of a classification model.

    Returns:
        Tuple[float, float, torch.Tensor]: The predicted label, its probability,
            and all the label probabilities.
    """
    sm = nn.Softmax(1)
    probs = sm(logits).squeeze()
    return probs.argmax().item(), probs.max().item(), probs


####################################################################################################
# functions regarding adversarial attacks                                                          #
####################################################################################################


def in_place_qr(A: torch.Tensor) -> torch.Tensor:
    """Computes the orthonormalisation of a torch.Tensor in place with the Gram-Schmidt algorithm.
    This makes it an in-place alternative to ```torch.linalg.qr```.
    Further, the function returns the relative amount of the squared norms of the orthogonal
    components that is due to their original vectors.

    Args:
    - A (torch.Tensor): The input tensor to be orthonormalised.

    Returns:
    - torch.Tensor: The relative amount of the squared norms of the orthogonal components that is due to their original vectors.
    """
    with torch.no_grad():
        device = A.device
        C = torch.zeros(len(A.T), device=device)
        D = torch.empty(len(A.T), device=device)

        # compute QR decomposition of A in-place by looping though its columns
        for i in range(len(A.T)):
            # store norm of current vector and normalise it
            D[i] = torch.linalg.norm(A.T[i])
            A.T[i] /= D[i]

            # compute orthogonal component of current vector in the remaining ones
            x, y = A.T[i + 1 :].size()
            v = A.T[i].clone()
            w = v @ A.T[i + 1 :].T
            C[i + 1 :] += w**2

            # subtract orthogonal component of current vector from remaining vectors
            A.T[i + 1 :] -= w.resize_(x, 1) * v.resize_(1, y)
        return D**2 / (C + D**2)


class MarginLoss(nn.Module):
    """Loss function that is used to execute an attack on a classification model.

    Args:
        x (torch.Tensor): The input tensor.
        y (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The mean of the difference between the input tensor and the target tensor.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(x - x[0, y.item()])


def pgd_attack(
    x: torch.Tensor,
    model: nn.Module,
    epsilon: float,
    step_size: float,
    loss_fn: nn.Module = MarginLoss(),
    target: int = None,
    manifold_projection: function = None,
    max_iter: int = 50,
    early_stopping: bool = True,
) -> torch.Tensor:
    """Performs a projected gradient descent attack on a given input tensor and classifier.

    Args:
        x (torch.Tensor): Input tensor to be attacked.
        model (nn.Module): Classifier to be attacked.
        epsilon (float): Maximum perturbation size.
        step_size (float): Step size for the gradient descent.
        loss_fn (nn.Module, optional): Loss function to be used. Defaults to MarginLoss().
        target (int, optional): Target class for targeted attack. Defaults to None.
        manifold_projection (function, optional): Function to project gradient onto a manifold. Defaults to None.
        max_iter (int, optional): Maximum number of iterations. Defaults to 50.
        early_stopping (bool, optional): Whether to stop early if attack is successful. Defaults to True.

    Returns:
        torch.Tensor: Perturbed input tensor.
    """
    x_ = x.clone().detach()  # keep original input unchanged
    device = x_.device
    assert epsilon > 0

    # define projection onto epsilon-ball around x
    def projection(t):
        d = torch.norm(t - x)
        if d <= epsilon:
            return t
        else:
            return epsilon * t / d + x * (1 - epsilon / d)

    # set pixel value ranges
    if x.size()[-1] == 224:  # detect ImageNet input
        mins = (
            torch.tensor([-2.11790393, -2.03571429, -1.80444444])
            .resize_(3, 1, 1)
            .to(device)
        )
        mins = torch.ones_like(x_) * mins
        maxs = torch.tensor([2.2489083, 2.42857143, 2.64]).resize_(3, 1, 1).to(device)
        maxs = torch.ones_like(x_) * maxs
    else:
        mins = -1.0
        maxs = +1.0

    # set attack mode
    targeted = bool(target)
    model.eval()

    with torch.no_grad():
        if not targeted:
            target = model(x).argmax(1)
        else:
            target = torch.tensor([target], device=device, dtype=torch.int)

    assert step_size > 0
    step_size = step_size if targeted else -1 * step_size

    # perform attack
    for i in range(max_iter):
        x_.requires_grad_(True)
        pred = model(x_)

        # check if attack is successful
        if early_stopping and (targeted ^ (pred.argmax(1).item() != target.item())):
            logging.info(f"pgd attack successful after {i} iterations")
            return x_.clone().detach()

        # compute gradient of x_
        loss = loss_fn(pred, target)
        model.zero_grad()
        assert x_.grad is None
        loss.backward()
        grad = x_.grad.data

        # update x_
        with torch.no_grad():
            grad = manifold_projection(grad) if manifold_projection else grad
            grad_norm = torch.norm(grad)

            # check if attack failed
            if grad_norm.item() == 0:
                logging.info("pgd attack not successful: gradient == 0")
                return x_.clone().detach()

            grad = grad / grad_norm
            x_ = x_ - step_size * grad
            x_ = torch.clamp(projection(x_), mins, maxs)

    logging.info("pgd attack reached max_iter")
    return x_.clone().detach()


def adv_attack_standard(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epsilon: float,
    step_size: float,
    device: torch.device,
    max_n: int = 10,
    max_iter: int = 50,
    loss_fct: nn.Module = MarginLoss(),
    target: int = None,
) -> list[tuple[torch.Tensor]]:
    """Performs a standard adversarial attack on a given model and dataset.

    Args:
        model (nn.Module): The model to be attacked.
        dataloader (torch.utils.data.DataLoader): The dataloader for the dataset to be attacked.
        epsilon (float): The maximum perturbation allowed for each pixel.
        step_size (float): The step size for each iteration of the attack.
        device (torch.device): The device to run the attack on.
        max_n (int, optional): The maximum number of adversarial examples to generate. Defaults to 10.
        max_iter (int, optional): The maximum number of iterations for each adversarial example. Defaults to 50.
        loss_fct (nn.Module, optional): The loss function to use for the attack. Defaults to MarginLoss().
        target (int, optional): The target class for the attack. If None, performs an untargeted attack. Defaults to None.

    Returns:
        list[tuple[torch.Tensor]]: A list of tuples containing the original input, the perturbed input, the model's prediction on the original input, and the model's prediction on the perturbed input.
    """
    assert epsilon > 0
    model.eval()
    examples = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # check if prediction is correct
        with torch.no_grad():
            pred_o = model(x)
            if pred_o.argmax(1).item() != y.item():
                continue

        # perform attack
        perturbed = pgd_attack(
            x,
            model,
            epsilon,
            step_size,
            max_iter=max_iter,
            loss_fn=loss_fct,
            target=target,
        )

        # check if attack was successful
        with torch.no_grad():
            pred_a = model(perturbed)
            if pred_a.argmax(1).item() != y.item():
                examples.append(
                    (
                        x.cpu().detach(),
                        perturbed.cpu().detach(),
                        pred_o.cpu(),
                        pred_a.cpu(),
                    )
                )
                logging.info(f"adversarial example {len(examples)} found")
                if len(examples) == max_n:
                    break

    return examples


def adv_attack_manifold(
    model: nn.Module,
    autoencoder: nn.Module,
    transform: nn.Module,
    inv_transform: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epsilon: tuple[float],
    step_size: tuple[float],
    device: torch.device,
    max_n: int = 10,
    max_iter: tuple[int] = (50, 1000, 1000),
    loss_fct: nn.Module = MarginLoss(),
    target: int = None,
) -> list[tuple[torch.Tensor]]:
    """Performs an adversarial attack on a given model and dataset using no/on/off manifold projection.

    Args:
    - model (nn.Module): The model to be attacked.
    - autoencoder (nn.Module): The autoencoder used for manifold projection.
    - transform (nn.Module): The transformation applied to the autoencoder output.
    - inv_transform (nn.Module): The inverse transformation applied to the input.
    - dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.
    - epsilon (tuple[float]): The maximum perturbation allowed for each attack.
    - step_size (tuple[float]): The step size for each attack.
    - device (torch.device): The device to perform the attack on.
    - max_n (int): The maximum number of adversarial examples to generate.
    - max_iter (tuple[int]): The maximum number of iterations for each attack.
    - loss_fct (nn.Module): The loss function to use for each attack.
    - target (int): The target class for targeted attacks.

    Returns:
    - examples (list[tuple[torch.Tensor]]): A list of tuples containing the original input, the perturbed input using standard attack, the perturbed input using on-manifold attack, the perturbed input using off-manifold attack, the original prediction, the prediction after standard attack, the prediction after on-manifold attack, and the prediction after off-manifold attack.
    """
    model.eval()
    autoencoder.eval()
    examples = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        s = x.size()
        assert s[0] == 1

        # check if prediction is correct
        with torch.no_grad():
            pred_o = model(x)
            if pred_o.argmax(1).item() != y.item():
                continue

        # perform standard attack
        perturbed = pgd_attack(
            x,
            model,
            epsilon[0],
            step_size[0],
            max_iter=max_iter[0],
            loss_fn=loss_fct,
            target=target,
        )

        # check if attack was successful
        with torch.no_grad():
            pred_a = model(perturbed)
            if pred_a.argmax(1).item() == y.item():
                continue

        # compute manifold projection
        logging.info("computing projection")
        with torch.no_grad():
            e = autoencoder.encoder(inv_transform(x))
        e.requires_grad_(True)
        d_ = autoencoder.decoder(e)
        d = transform(d_)
        n_pixels = d.numel()
        n_latents = e.numel()
        g = torch.empty(n_pixels, n_latents, device=device)

        # loop though output pixels to compute their jacobian w.r.t. the latent vector
        for i in range(n_pixels):
            autoencoder.zero_grad()
            transform.zero_grad()
            if i > 0:
                e.grad *= 0.0
            torch.flatten(d)[i].backward(retain_graph=(i < n_pixels - 1))
            g[i] = torch.flatten(e.grad)

        # orthonormalise the jacobian to get the projection matrix
        with torch.no_grad():
            in_place_qr(g)

        # define projection onto the manifold
        def projection_on(grad):
            grad = torch.flatten(grad)
            grad = g.T @ grad
            grad = g @ grad
            grad.resize_(s)
            return grad

        # define projection off the manifold
        def projection_off(grad):
            return grad - projection_on(grad)

        # perform on manifold attack
        perturbed_on = pgd_attack(
            x,
            model,
            epsilon[1],
            step_size[1],
            max_iter=max_iter[1],
            loss_fn=loss_fct,
            manifold_projection=projection_on,
            target=target,
        )

        # check if attack was successful
        with torch.no_grad():
            pred_a_on = model(perturbed_on)
            if pred_a_on.argmax(1).item() == y.item():
                continue

        # perform off manifold attack
        perturbed_off = pgd_attack(
            x,
            model,
            epsilon[2],
            step_size[2],
            max_iter=max_iter[2],
            loss_fn=loss_fct,
            manifold_projection=projection_off,
            target=target,
        )

        # check if attack was successful
        with torch.no_grad():
            pred_a_off = model(perturbed_off)
            if pred_a_off.argmax(1).item() == y.item():
                continue

        examples.append(
            (
                x.cpu().detach(),
                perturbed.cpu().detach(),
                perturbed_on.cpu().detach(),
                perturbed_off.cpu().detach(),
                pred_o.cpu(),
                pred_a.cpu(),
                pred_a_on.cpu(),
                pred_a_off.cpu(),
            )
        )
        logging.info(f"adversarial example {len(examples)} found")
        if len(examples) == max_n:
            break

    return examples

import torch
from torch import nn

####################################################################################################
# MNIST models
####################################################################################################


class MnistMLP(nn.Module):
    """A multi-layer perceptron (MLP) neural network for classifying images of handwritten digits from the MNIST dataset.

    Args:
        None

    Attributes:
        flatten (nn.Flatten): A PyTorch module that flattens the input tensor into a 1D tensor.
        linear (nn.Sequential): A PyTorch module that consists of three fully connected layers with ReLU activation and dropout regularization.

    Methods:
        forward(x): Performs a forward pass through the network.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, 1, 28, 28).

            Returns:
                logits (torch.Tensor): A tensor of shape (batch_size, 10) containing the predicted logits for each class.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 10),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class MnistAEC(nn.Module):
    """A convolutional autoencoder for the MNIST dataset.

    Args:
        track_running_stats (bool): Whether to track the running mean and variance
            of the batch normalization layer. Default is True.

    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder.
        decoder (nn.Sequential): The decoder part of the autoencoder.

    Methods:
        forward(x): Performs a forward pass through the network.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, 1, 28, 28).

            Returns:
                torch.Tensor: The output tensor of shape (batch_size, 1, 28, 28).
    """

    def __init__(self, track_running_stats: bool = True) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=128, track_running_stats=track_running_stats),
            nn.Softplus(beta=100),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256, track_running_stats=track_running_stats),
            nn.Softplus(beta=100),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512, track_running_stats=track_running_stats),
            nn.Softplus(beta=100),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=1024, track_running_stats=track_running_stats),
            nn.Softplus(beta=100),
            nn.Flatten(),
            nn.Linear(1024, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 16384),
            nn.Softplus(beta=100),
            nn.Unflatten(1, (1024, 4, 4)),
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512, track_running_stats=track_running_stats),
            nn.Softplus(beta=100),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256, track_running_stats=track_running_stats),
            nn.Softplus(beta=100),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=128, track_running_stats=track_running_stats),
            nn.Softplus(beta=100),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=2
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        e = self.encoder(x)
        d = self.decoder(e)
        return d


####################################################################################################
# CIFAR classifier
####################################################################################################


def CifarCNN(n_channel: int = 128) -> nn.Module:
    """Creates a convolutional neural network model for the CIFAR-10 dataset.

    Args:
        n_channel (int): Number of channels in the first convolutional layer. Default is 128.

    Returns:
        torch.nn.Module: A PyTorch module representing the CIFAR-10 CNN model.
    """
    cfg = [
        n_channel,
        n_channel,
        "M",
        2 * n_channel,
        2 * n_channel,
        "M",
        4 * n_channel,
        4 * n_channel,
        "M",
        (8 * n_channel, 0),
        "M",
    ]
    layers = _make_layers(cfg, batch_norm=True)
    model = _CIFAR(layers, n_channel=8 * n_channel, num_classes=10)
    return model


class _CIFAR(nn.Module):
    """A neural network model for CIFAR classification.

    Args:
        features (nn.Sequential): The convolutional feature extractor.
        n_channel (int): The number of output channels from the feature extractor.
        num_classes (int): The number of classes to predict.

    Attributes:
        features (nn.Sequential): The convolutional feature extractor.
        classifier (nn.Sequential): The fully connected classifier.

    Methods:
        forward(x): Forward pass through the network.

    """

    def __init__(
        self, features: nn.Sequential, n_channel: int, num_classes: int
    ) -> None:
        super().__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(n_channel, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    """Constructs a list of convolutional and pooling layers based on the given configuration.

    Args:
        cfg (list): A list of integers and/or tuples representing the configuration of the layers.
            If an integer is given, it represents the number of output channels for a convolutional layer.
            If a tuple is given, it represents the number of output channels and the padding size for a convolutional layer.
            If the string "M" is given, a max pooling layer is added to the list.
        batch_norm (bool, optional): Whether to use batch normalization after each convolutional layer. Defaults to False.

    Returns:
        nn.Sequential: A sequential container of the convolutional and pooling layers.
    """
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=padding
            )
            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(out_channels, affine=False),
                    nn.ReLU(),
                ]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)


class CifarAEC(nn.Module):
    """A convolutional autoencoder for CIFAR-10 images.

    The autoencoder consists of an encoder and a decoder. The encoder takes in a CIFAR-10 image
    and produces a compressed representation of the image. The decoder takes in the compressed
    representation and produces a reconstructed image.

    Attributes:
        encoder (nn.Sequential): The encoder network.
        decoder (nn.Sequential): The decoder network.

    Methods:
        forward(x): Passes the input tensor through the encoder and decoder networks.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, 3, 32, 32).

            Returns:
                torch.Tensor: The reconstructed tensor of shape (batch_size, 3, 32, 32).
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.Softplus(beta=100),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.Softplus(beta=100),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.Softplus(beta=100),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.Softplus(beta=100),
            nn.Flatten(),
            nn.Linear(4096, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 16384),
            nn.Softplus(beta=100),
            nn.Unflatten(1, (1024, 4, 4)),
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.Softplus(beta=100),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.Softplus(beta=100),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.Softplus(beta=100),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> None:
        e = self.encoder(x)
        d = self.decoder(e)
        return d


####################################################################################################
# VGG16
####################################################################################################


class VGG16(nn.Module):
    """A VGG16-based autoencoder model.

    Args:
        n_classes (int): The number of classes to predict.
        in_channels (int): The number of input channels.

    Attributes:
        down1 (segnetDown2): The first down block.
        down2 (segnetDown2): The second down block.
        down3 (segnetDown3): The third down block.
        down4 (segnetDown3): The fourth down block.
        down5 (segnetDown3): The fifth down block.
        up5 (segnetUp3): The first up block.
        up4 (segnetUp3): The second up block.
        up3 (segnetUp3): The third up block.
        up2 (segnetUp2): The fourth up block.
        up1 (segnetUp2): The fifth up block.

    Methods:
        forward(inputs): Defines the computation performed at every call.
        encoder(inputs): Returns the output of the encoder.
        decoder(inputs): Returns the output of the decoder.
    """


class VGG16(nn.Module):
    def __init__(self, n_classes: int = 3, in_channels: int = 3) -> None:
        super(VGG16, self).__init__()

        self.down1 = segnetDown2(in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes, sigmoid=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        up5 = self.up5(down5)
        up4 = self.up4(up5)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)

        return up1

    def encoder(self, inputs: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        return down5

    def decoder(self, inputs: torch.Tensor) -> torch.Tensor:
        up5 = self.up5(inputs)
        up4 = self.up4(up5)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        return up1


class segnetDown2(nn.Module):
    """Downsampling block for SegNet architecture with 2 layers.

    Args:
        in_size (int): Number of input channels.
        out_size (int): Number of output channels.

    Methods:
        forward(x): Passes the input tensor through the layers.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, in_size, H, W).

            Returns:
                torch.Tensor: The output tensor of shape (batch_size, out_size, H/2, W/2).
    """

    def __init__(self, in_size: int, out_size: int) -> None:
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.maxpool(outputs)
        return outputs


class segnetDown3(nn.Module):
    """Downsampling block for SegNet architecture with 3 layers.

    Args:
        in_size (int): Number of input channels.
        out_size (int): Number of output channels.

    Methods:
        forward(x): Passes the input tensor through the layers.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, in_size, H, W).

            Returns:
                torch.Tensor: The output tensor of shape (batch_size, out_size, H/2, W/2).
    """


def __init__(self, in_size: int, out_size: int) -> None:
    super(segnetDown3, self).__init__()
    self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
    self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
    self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
    self.maxpool = nn.MaxPool2d(2, 2)


def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    outputs = self.maxpool(outputs)
    return outputs


class segnetUp2(nn.Module):
    """Upsampling block for SegNet architecture with 2 layers.

    Args:
        in_size (int): Number of input channels.
        out_size (int): Number of output channels.
        sigmoid (bool): whether to add a sigmoid activation after the last layer.

        Methods:
        forward(x): Passes the input tensor through the layers.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, in_size, H, W).

            Returns:
                torch.Tensor: The output tensor of shape (batch_size, out_size, 2*H, 2*W).
    """

    def __init__(self, in_size: int, out_size: int, sigmoid: bool = False) -> None:
        super(segnetUp2, self).__init__()
        self.unpool = nn.Upsample(scale_factor=2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1, sigmoid=sigmoid)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.unpool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    """Upsampling block for SegNet architecture with 3 layers.

    Args:
        in_size (int): Number of input channels.
        out_size (int): Number of output channels.

        Methods:
        forward(x): Passes the input tensor through the layers.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, in_size, H, W).

            Returns:
                torch.Tensor: The output tensor of shape (batch_size, out_size, 2*H, 2*W).
    """

    def __init__(self, in_size: int, out_size: int) -> None:
        super(segnetUp3, self).__init__()
        self.unpool = nn.Upsample(scale_factor=2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.unpool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    """A class that implements a 2D convolutional layer followed by batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of output channels.
        k_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding of the convolution.
        bias (bool, optional): Whether to include a bias term in the convolution. Default is True.
        dilation (int or tuple, optional): Dilation rate of the convolution. Default is 1.
        sigmoid (bool, optional): Whether to use sigmoid activation instead of ReLU. Default is False.

    Methods:
        forward(x): Passes the input tensor through the convolutional layer, batch normalization, and ReLU activation.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        k_size: int or tuple,
        stride: int or tuple,
        padding: int or tuple,
        bias: bool = True,
        dilation: int or tuple = 1,
        sigmoid: bool = False,
    ) -> None:
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if sigmoid:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.Sigmoid()
            )
        else:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU()
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.cbr_unit(inputs)
        return outputs


####################################################################################################
# isometric autoencoder losses
####################################################################################################


class IsoLoss(nn.Module):
    """Loss function to compute the Isometry loss of the Isometric Autoencoder from [1].

    Args:
        lam (float): The weight of the loss term.
        device (torch.device): The device to use for computations.
        lat_dims (int, optional): The number of latent dimensions. Defaults to 2.
        out_dims (int, optional): The number of output dimensions. Defaults to 4.

    Attributes:
        lam (torch.Tensor): The weight of the loss term.
        mse (nn.MSELoss): The mean squared error loss function.
        lat (int): The number of latent dimensions.
        out (int): The number of output dimensions.
        device (torch.device): The device to use for computations.

    Methods:
        forward(jacobian: torch.Tensor) -> torch.Tensor:
            Computes the Isometry loss given the Jacobian matrix.

                Args:
                jacobian (torch.Tensor): The Jacobian matrix of the model.

                Returns:
                    torch.Tensor: The Isometry loss.

    References:
        [1] Isometric Autoencoders for Learning Disentangled Representations
            https://arxiv.org/abs/2006.09289
    """

    def __init__(
        self,
        lam: float,
        device: torch.device,
        lat_dims: int = 2,  # size (batch, latent_features)
        out_dims: int = 4,  # size (batch, channels, height, width)
    ) -> None:
        super().__init__()
        self.lam = torch.tensor(lam).to(device)
        self.mse = nn.MSELoss().to(device)
        self.lat = lat_dims
        self.out = out_dims
        self.device = device

    def forward(self, jacobian: torch.Tensor) -> torch.Tensor:
        assert jacobian.dim() == 1 + self.lat + self.out

        # flatten jacobian to (batch_size, output_size, latent_size)
        d = torch.flatten(
            torch.flatten(jacobian, start_dim=-self.lat), start_dim=1, end_dim=self.out
        )

        assert d.dim() == 3
        s = d.size()

        # compute loss
        u = torch.randn(s[0], s[2], 1).to(self.device)
        u = u / torch.linalg.vector_norm(u, dim=1, keepdim=True)
        du = torch.matmul(d, u)
        assert du.dim() == 3

        l = self.lam * self.mse(du, torch.ones_like(du, device=self.device))
        return l


class PIsoLoss(nn.Module):
    """Loss function to compute the Pseudo-Inverse Isometry loss of the Isometric Autoencoder from [1].

    Args:
        lam (float): The weight of the loss term.
        device (torch.device): The device to use for computations.
        lat_dims (int, optional): The number of latent dimensions. Defaults to 2.
        out_dims (int, optional): The number of output dimensions. Defaults to 4.

    Attributes:
        lam (torch.Tensor): The weight of the loss term.
        mse (nn.MSELoss): The mean squared error loss function.
        lat (int): The number of latent dimensions.
        out (int): The number of output dimensions.
        device (torch.device): The device to use for computations.

    Methods:
        forward(jacobian: torch.Tensor) -> torch.Tensor:
            Computes the Pseudo-Inverse Isometry loss given the Jacobian matrix.

                Args:
                jacobian (torch.Tensor): The Jacobian matrix of the model.

                Returns:
                    torch.Tensor: The Isometry loss.

    References:
        [1] Isometric Autoencoders for Learning Disentangled Representations
            https://arxiv.org/abs/2006.09289
    """

    def __init__(
        self,
        lam: float,
        device: torch.device,
        in_dims: int = 4,  # size (batch, channels, height, width)
        lat_dims: int = 2,  # size (batch, latent_features)
    ) -> None:
        super().__init__()
        self.lam = torch.tensor(lam).to(device)
        self.mse = nn.MSELoss().to(device)
        self.in_ = in_dims
        self.lat = lat_dims
        self.device = device

    def forward(self, jacobian: torch.Tensor) -> torch.Tensor:
        assert jacobian.dim() == 1 + self.in_ + self.lat

        # flatten jacobian to (batch_size, latent_size, input_size)
        d = torch.flatten(
            torch.flatten(jacobian, start_dim=-self.in_), start_dim=1, end_dim=self.lat
        )

        assert d.dim() == 3
        s = d.size()

        # compute loss
        uT = torch.randn(s[0], 1, s[1]).to(self.device)
        uT = uT / torch.linalg.vector_norm(uT, dim=2, keepdim=True)
        uTd = torch.matmul(uT, d)
        assert uTd.dim() == 3

        l = self.lam * self.mse(uTd, torch.ones_like(uTd, device=self.device))
        return l

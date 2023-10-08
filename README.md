# The Dimpled Manifold Revisited

This is the repository of my university research project reviewing the findings of the paper [1] by Shamir et.al. which proposes a new explanation for the phenomenon of adversarial attacks.

The purpose of this project was to reproduce the results of [1], and, based on the insights they provide, come up with a new defence against adversarial attacks on neural network classifiers.
While the first objective could largely be achieved, the second objective proved to be quite challenging and could not be fullfilled in the given timeframe of the project.

Detailed information about the motivation of the project and its results can be found on the poster below and in the slides in the [```documents```](https://github.com/LukasKarner/dimpled-manifolds/tree/main/documents) directory.


The datasets used for this project were MNIST, CIFAR-10, and ImageNet.
The implementation is based purely on PyTorch, the code for training and evaluating the classifiers and autoencoders, for computing the adversarial attacks, and for conducting the observations related to the manifold structure can be found in the respective directories.
Due to the large file sizes, the models and data are not included in this repository, however, I can send them to you upon request.

[1] ["The Dimpled Manifold Model of Adversarial Examples in Machine Learning"](https://arxiv.org/abs/2106.10151)

## Poster

![poster](documents/poster.jpg)

## Table of Content
> * [Deep Convolutional Generative Adversarial Networks (DCGANs) - Pytorch](#DeepConvolutionalGenerativeAdversarialNetworks(GANs)-Pytorch)
>   * [About the Project](#AbouttheProject)
>   * [About Database](#AboutDatabases)
>   * [Built with](#Builtwith)
>   * [Installation](#Installation)
>   * [Examples](#Example)

# Deep Convolutional Generative Adversarial Networks (DCGANs) - Pytorch
## About the Project
This project focuses on develop the Deep Convolutional GAN (DCGAN) to generate new images similar to the STL-10 dataset using PyTorch.

![Gan](https://user-images.githubusercontent.com/75105778/153688204-0a4fdaae-d7c0-44b8-b3c2-e95b0185e04d.jpg)

The generator generates fake data and the discriminator identifies real images from fake images. The generator and the discriminator compete with each other in a game in the training stage. This competition is generating better-looking images to deceive the discriminator by the generator and getting better at identifying real images from fake images by the discriminator.

Steps of Developing This project:

![Gan-recipes](https://user-images.githubusercontent.com/75105778/153688224-873dae4f-8d2f-4ede-aae4-c58d1397dff0.jpg)


## About Database

Dataset is STL-10 dataset from the PyTorch torchvision package.

For more information about  https://cs.stanford.edu/~acoates/stl10 .


## Built with
* Pytorch
* Binary cross-entropy (BCE) loss function for both of generator and discriminator.
* Adam optimizer.

## Installation
    •	conda install pytorch torchvision cudatoolkit=coda version -c pytorch

## Examples


![Gan-result](https://user-images.githubusercontent.com/75105778/153688364-7b04e260-0f42-453b-af01-34dbd3bf2016.png)


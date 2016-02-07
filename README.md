# Discriminative Regularization for Generative Models
Code for the _Discriminative Regularization for Generative Models_ paper.

## Requirements

* [Blocks](https://blocks.readthedocs.org/en/latest/), development version
* [Fuel](https://fuel.readthedocs.org/en/latest/), development version

## Downloading and converting the datasets

Set up your `~/.fuelrc` file:

``` bash
$ echo "data_path: \"<MY_DATA_PATH>\"" > ~/.fuelrc
```

Go to `<MY_DATA_PATH>`:

``` bash
$ cd <MY_DATA_PATH>
```

Download the SVHN format 2 dataset:

``` bash
$ fuel-download svhn 2
$ fuel-convert svhn 2
$ fuel-download svhn 2 --clear
```

Download the CIFAR-10 dataset:

``` bash
$ fuel-download cifar10
$ fuel-convert cifar10
$ fuel-download cifar10 --clear
```

Download the CelebA dataset:

``` bash
$ fuel-download celeba 64
$ fuel-convert celeba 64
$ fuel-download celeba 64 --clear
```

## Training the models

Make sure you're in the repo's root directory.

### SVHN

**WRITEME**

### CIFAR-10

**WRITEME**

### CelebA

Train the CelebA classifier:

``` bash
$ THEANORC=theanorc python train_celeba_classifier.py
```

Train a VAE *without* discriminative regularization:

``` bash
$ THEANORC=theanorc python train_celeba_vae.py
```

Train a VAE *with* discriminative regularization:

``` bash
$ THEANORC=theanorc python train_celeba_vae.py --regularize
```

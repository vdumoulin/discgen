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

### Download VGG19

Required for the SVHN and CIFAR10 models.

``` bash
python scripts/download_vgg19
```

### SVHN

Make sure you downloaded VGG19.

Train a VAE *without* discriminative regularization:

``` bash
$ THEANORC=theanorc python experiments/train_svhn_vae.py
```

Train a VAE *with* discriminative regularization:

``` bash
$ THEANORC=theanorc python experiments/train_svhn_vae.py --regularize
```

### CIFAR-10

Make sure you downloaded VGG19.

Train a VAE *without* discriminative regularization:

``` bash
$ THEANORC=theanorc python experiments/train_cifar10_vae.py
```

Train a VAE *with* discriminative regularization:

``` bash
$ THEANORC=theanorc python experiments/train_cifar10_vae.py --regularize
```

### CelebA

Train the CelebA classifier:

``` bash
$ THEANORC=theanorc python experiments/train_celeba_classifier.py
```

Train a VAE *without* discriminative regularization:

``` bash
$ THEANORC=theanorc python experiments/train_celeba_vae.py
```

Train a VAE *with* discriminative regularization:

``` bash
$ THEANORC=theanorc python experiments/train_celeba_vae.py --regularize
```

## Evaluating the models

### Samples

``` bash
$ THEANORC=theanorc scripts/sample [trained_model.zip]
```

### Reconstructions

``` bash
$ THEANORC=theanorc scripts/reconstruct [which_dataset] [trained_model.zip]
```

### Interpolations

``` bash
$ THEANORC=theanorc scripts/interpolate [which_dataset] [trained_model.zip]
```

### NLL approximation

``` bash
$ THEANORC=theanorc scripts/compute_nll_approximation [which_dataset] [trained_model.zip]
```

*Note: this takes a __long__ time.*

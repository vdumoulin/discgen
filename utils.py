"""Utility functions."""
import six
from blocks.bricks import Rectifier
from blocks.bricks.conv import (ConvolutionalSequence, Convolutional,
                                AveragePooling)
from blocks.initialization import Constant
from fuel.datasets import SVHN, CIFAR10, CelebA
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid
from six.moves import zip, cPickle


def plot_image_grid(images, num_rows, num_cols, save_path=None):
    """Plots images in a grid.

    Parameters
    ----------
    images : numpy.ndarray
        Images to display, with shape
        ``(num_rows * num_cols, num_channels, height, width)``.
    num_rows : int
        Number of rows for the image grid.
    num_cols : int
        Number of columns for the image grid.
    save_path : str, optional
        Where to save the image grid. Defaults to ``None``,
        which causes the grid to be displayed on screen.

    """
    figure = pyplot.figure()
    grid = ImageGrid(figure, 111, (num_rows, num_cols), axes_pad=0.1)

    for image, axis in zip(images, grid):
        axis.imshow(image.transpose(1, 2, 0), interpolation='nearest')
        axis.set_yticklabels(['' for _ in range(image.shape[1])])
        axis.set_xticklabels(['' for _ in range(image.shape[2])])
        axis.axis('off')

    if save_path is None:
        pyplot.show()
    else:
        pyplot.savefig(save_path, transparent=True, bbox_inches='tight')


def create_streams(train_set, valid_set, test_set, training_batch_size,
                   monitoring_batch_size):
    """Creates data streams for training and monitoring.

    Parameters
    ----------
    train_set : :class:`fuel.datasets.Dataset`
        Training set.
    valid_set : :class:`fuel.datasets.Dataset`
        Validation set.
    test_set : :class:`fuel.datasets.Dataset`
        Test set.
    monitoring_batch_size : int
        Batch size for monitoring.
    include_targets : bool
        If ``True``, use both features and targets. If ``False``, use
        features only.

    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.

    """
    main_loop_stream = DataStream.default_stream(
        dataset=train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, training_batch_size))
    train_monitor_stream = DataStream.default_stream(
        dataset=train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, monitoring_batch_size))
    valid_monitor_stream = DataStream.default_stream(
        dataset=valid_set,
        iteration_scheme=ShuffledScheme(
            valid_set.num_examples, monitoring_batch_size))
    test_monitor_stream = DataStream.default_stream(
        dataset=test_set,
        iteration_scheme=ShuffledScheme(
            test_set.num_examples, monitoring_batch_size))

    return (main_loop_stream, train_monitor_stream, valid_monitor_stream,
            test_monitor_stream)


def create_svhn_streams(training_batch_size, monitoring_batch_size):
    """Creates SVHN data streams.

    Parameters
    ----------
    training_batch_size : int
        Batch size for training.
    monitoring_batch_size : int
        Batch size for monitoring.

    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.

    """
    train_set = SVHN(2, ('train',), sources=('features',),
                     subset=slice(0, 63257))
    valid_set = SVHN(2, ('train',), sources=('features',),
                     subset=slice(63257, 73257))
    test_set = SVHN(2, ('test',), sources=('features',))

    return create_streams(train_set, valid_set, test_set, training_batch_size,
                          monitoring_batch_size)


def create_cifar10_streams(training_batch_size, monitoring_batch_size):
    """Creates CIFAR10 data streams.

    Parameters
    ----------
    training_batch_size : int
        Batch size for training.
    monitoring_batch_size : int
        Batch size for monitoring.

    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.

    """
    train_set = CIFAR10(('train',), sources=('features',),
                     subset=slice(0, 45000))
    valid_set = CIFAR10(('train',), sources=('features',),
                     subset=slice(45000, 50000))
    test_set = CIFAR10(('test',), sources=('features',))

    return create_streams(train_set, valid_set, test_set, training_batch_size,
                          monitoring_batch_size)


def create_celeba_streams(training_batch_size, monitoring_batch_size,
                          include_targets=False):
    """Creates CelebA data streams.

    Parameters
    ----------
    training_batch_size : int
        Batch size for training.
    monitoring_batch_size : int
        Batch size for monitoring.
    include_targets : bool
        If ``True``, use both features and targets. If ``False``, use
        features only.

    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.

    """
    sources = ('features', 'targets') if include_targets else ('features',)

    train_set = CelebA('64', ('train',), sources=sources)
    valid_set = CelebA('64', ('valid',), sources=sources)
    test_set = CelebA('64', ('test',), sources=sources)

    return create_streams(train_set, valid_set, test_set, training_batch_size,
                          monitoring_batch_size)


def load_vgg_classifier():
    """Loads the VGG19 classifier into a brick.

    Relies on ``vgg19_normalized.pkl`` containing the model
    parameters.

    Returns
    -------
    convnet : :class:`blocks.bricks.conv.ConvolutionalSequence`
        VGG19 convolutional brick.

    """
    convnet = ConvolutionalSequence(
        layers=[
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=64,
                name='conv1_1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=64,
                name='conv1_2'),
            Rectifier(),
            AveragePooling(
                pooling_size=(2, 2),
                name='pool1'),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=128,
                name='conv2_1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=128,
                name='conv2_2'),
            Rectifier(),
            AveragePooling(
                pooling_size=(2, 2),
                name='pool2'),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv3_1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv3_2'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv3_3'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv3_4'),
            Rectifier(),
            AveragePooling(
                pooling_size=(2, 2),
                name='pool3'),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv4_1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv4_2'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv4_3'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv4_4'),
            Rectifier(),
            AveragePooling(
                pooling_size=(2, 2),
                name='pool4'),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv5_1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv5_2'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv5_3'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv5_4'),
            Rectifier(),
            AveragePooling(
                pooling_size=(2, 2),
                name='pool5'),
        ],
        num_channels=3,
        image_size=(32, 32),
        tied_biases=True,
        weights_init=Constant(0),
        biases_init=Constant(0),
        name='convnet')
    convnet.initialize()

    with open('vgg19_normalized.pkl', 'rb') as f:
        if six.PY3:
            data = cPickle.load(f, encoding='latin1')
        else:
            data = cPickle.load(f)
        parameter_values = data['param values']
    conv_weights = parameter_values[::2]
    conv_biases = parameter_values[1::2]
    conv_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
    conv_layers = [convnet.layers[i] for i in conv_indices]
    for layer, W_val, b_val in zip(conv_layers, conv_weights, conv_biases):
        W, b = layer.parameters
        W.set_value(W_val)
        b.set_value(b_val)

    return convnet

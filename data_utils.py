"""Contains utilities for downloading and converting datasets."""

import os
import sys
from six.moves import urllib
import zipfile
import random

import tensorflow as tf

# The default file where class_names and corresponding ids are stored.
LABELS_FILENAME = 'labels.txt'


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, label, image_shape):
    """ Serialize features into string
    Args:
        image: Image in bytes
        label:  Label in 0-based scalar
        image_shape: Image shape in (height, width, nchannel)
    Returns:
        Reture the serialized string for whole inputs
    """
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
    }

    #  Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def read_tfrecord(serialized_example):
    """ Transform serialized string to designated data
    Args:
        serialized_example: Serialized string to be decoded
    Returns:
        A tuple of decoded image and its 0-based label
    """
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type = float)
    image_shape = [example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)
    label = example['label']
    
    return image, label


def download_and_uncompress(zip_url, local_filepath, uncompressed_dir):
    """Downloads the `zip_url` and uncompresses it locally.
    Args:
        zip_url: The URL of a zip file.
        local_filepath: The local path where the downloaded is stored.
        uncompressed_dir: The directory where the temporary files are stored.
    """
    # The callback function to display the dowload progress
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            local_filepath, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    
    # Retrieve the zipfile from designated location, in this case, file:// protocal is used
    local_filepath, _ = urllib.request.urlretrieve(zip_url, local_filepath, _progress)
    print()
    statinfo = os.stat(local_filepath)
    print('Successfully downloaded', local_filepath, statinfo.st_size, 'bytes.')
    
    # Uncompressed the zip file to designated location
    zfile = zipfile.ZipFile(local_filepath)
    zfile.extractall(uncompressed_dir)

    
def split_dataset(filenames, split={'train':0.7, 'dev': 0.15, 'test': 0.15}):
    """ Shffule to split dataset to train, dev, and test. Default is 0.7:0.15:0.15
    Args:
        filenames: List of filenames to be shuffled and splitted
        split: The ratio to be splitted
    Returns:
        A dictionary indexed by splitted set name and corresponding filenames
    """
    # Sort and shuffle after to make sure the order of filename is reproducible
    filenames.sort()
    random.shuffle(filenames)
    
    # Store corresponding filenames to its datasets
    split_filenames = {}
    accum_ratio, accum_idx = 0.0, 0
    num_filenames = len(filenames)
    for k, r in split.items():
        accum_ratio += r
        split_idx = int(accum_ratio * num_filenames)
        split_filenames[k] = filenames[accum_idx:split_idx]
        accum_idx = split_idx

    return split_filenames


def write_label_file(labels_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    """ Write dictionary of class label to id mapping information to file
    Args:
        labels_to_class_names: The dictionary mapping for label name to id
        dataset_dir: The folder to save label file
        filename: File to be stored
    Returns:
        None
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write("%d:%s\n" % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """ Examine if the given file exist in folder
    Args:
        dataset_dir: The directory to check file existence.
        filename: The label file name.
    Returns:
        `True` if the labels file exists and `False` otherwise.
    """
    return os.path.exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """ Reads the labels file and returns a mapping from ID to class name.
    Args:
        dataset_dir: The directory in which the labels file is found.
        filename: The filename where the class names are written.
    Returns:
        A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with open(labels_filename, 'r') as f:
        lines = f.read()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names
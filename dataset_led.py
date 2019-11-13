"""Contains methods for loading data in various way."""

import tensorflow as tf

import download_and_convert as dc
import data_utils

# The default directory where the temporary files and the TFRecords are stored.
_DATA_DIR = 'data'


def load_data_numpy():
    """ Get the data in ndarray format
    Args:
        None
    Returns:
        A tuple of train, validatioin and test dataset in ndarray format
    """
    # Not implemented 
    return

def load_data_tfdataset():
    """ Get the data in tfdataset format
    Args:
        None
    Returns:
        A tuple of train, validatioin and test dataset in tfrecord format
    """
    # Get and unzip the compressed file to designated folder if necessary
    # This will transform all the original data into tfrecord format
    dc.run(_DATA_DIR)
    
    # Load train dataset and decode to tensors
    train_tfrecords = tf.data.TFRecordDataset(dc.get_tfrecord_filename(_DATA_DIR, 'train'))
    parsed_train = train_tfrecords.map(data_utils.read_tfrecord)

    # Load validation dataset and decode to tensors
    dev_tfrecords = tf.data.TFRecordDataset(dc.get_tfrecord_filename(_DATA_DIR, 'dev'))
    parsed_dev = dev_tfrecords.map(data_utils.read_tfrecord)

    # Load test dataset and decode to tensors
    test_tfrecords = tf.data.TFRecordDataset(dc.get_tfrecord_filename(_DATA_DIR, 'test'))
    parsed_test = test_tfrecords.map(data_utils.read_tfrecord)

    return parsed_train, parsed_dev, parsed_test
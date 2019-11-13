"""Contains main script for downloading and converting datasets."""

import os
import sys
import random
import shutil

import tensorflow as tf

import data_utils

# The URL where the data can be downloaded.
_DATA_URL = 'file:///home/hankkuo/PowerArena/LED-Digits-Recognition/dataset.zip'

# The dataset name, which can be used in all the prefix of processed filename
_DATA_NAME = _DATA_URL.split('/')[-1]


def get_tfrecord_filename(dataset_dir, split_name):
    """ Get the tfrecord filename specified by split type
    Args:
        dataset_dir: The folder stored tfrecord files
        split_name: The split name of the tfrecord, usually train, dev and test
    Returns:
        A string of path for the tfrecord (name is combined with split type)
    """
    base_filename = os.path.splitext(os.path.basename(_DATA_URL))[0]
    return os.path.join(dataset_dir, base_filename + '_' + split_name + '.tfrecord') 


def saparate_dataset(uncompressed_dir, split_data_dir, ratio = {'train':0.7, 'dev': 0.15, 'test': 0.15}):
    """ Saparate the files from original folder to split folder, eq. train, dev and test. The amount of
        files would follow the designated ratio and stay in corresponding class folder
    Args:
        uncompressed_dir: The folder stores folder named by class
        split_data_dir: The folder to store the saparated folders, train, dev and test
    Returns:
        None
    """
    # Check if split destination exists
    if not tf.io.gfile.exists(split_data_dir):
        tf.io.gfile.mkdir(split_data_dir)
    
    #  Make dir for specified saparation
    for k in ratio:
        dir_path = os.path.join(split_data_dir, k)
        if not tf.io.gfile.exists(dir_path):
            tf.io.gfile.mkdir(dir_path)

    # Make the shuffle reproducible
    random.seed(523)
    
    # Walkthrough dirs under uncompressed and split them into saparation dir
    for i, dir_ in enumerate(os.listdir(uncompressed_dir)):
        dir_path = os.path.join(uncompressed_dir, dir_)
        filenames = os.listdir(dir_path)

        sys.stdout.write("\r>>Saparating images into sets %d/%d" %
                         (i+1, len(os.listdir(uncompressed_dir))))
        sys.stdout.flush()

        split_filenames = data_utils.split_dataset(filenames, ratio)
        for split, filelist in split_filenames.items():
            split_class_path = os.path.join(split_data_dir, os.path.join(split, dir_))
            if not tf.io.gfile.exists(split_class_path):
                tf.io.gfile.mkdir(split_class_path)
            src_path = map(lambda x: os.path.join(dir_path, x), filelist) 
            dst_path = map(lambda x: os.path.join(split_class_path, x), filelist)
            for src, dst in zip(src_path, dst_path):
                shutil.copyfile(src, dst)

    sys.stdout.write('\n')
    sys.stdout.flush()


def get_classnames(dataset_dir):
    """ Get class name from the folder name under specified path
    Args:
        dataset_dir: The location to store all the folder named by class name
    Returns:
        Class name collected from folder name
    """
    class_names = []
    for name in os.listdir(dataset_dir):
        d_path = os.path.join(dataset_dir, name)
        if os.path.isdir(d_path):
            class_names.append(name)
    return class_names


def get_samples(dataset_dir, class_names_to_ids):
    """ List all the sample under given dir and its corresponding class id
    Args:
        dataset_dir: The location where the folder named by class name are stored
        class_names_to_ids: A dictionary mapping for class name to ids
    Returns:
        list of tuples represent filename and corresponding id
    """
    samples = []
    for d in os.listdir(dataset_dir):
        d_path = os.path.join(dataset_dir, d)
        if os.path.isdir(d_path):
            for filename in os.listdir(d_path):
                filepath = os.path.join(d_path, filename)
                samples.append((filepath, class_names_to_ids[d]))
    return samples


def convert_dataset(samples, output_tfrecord):
    """ Main function to convert samples to tfrecord
    Args:
        samples: List of tuple for sample (filename, id)
        output_tfrecord: The file to store serialized samples
    Returns:
        None
    """    
    with tf.io.TFRecordWriter(output_tfrecord) as tfrecord_writer:
        for i, sample in enumerate(samples):
            try:
                sys.stdout.write("\r>>Converting image into %s %d/%d" % (output_tfrecord, i+1, len(samples)))
                sys.stdout.flush()

                filename, class_id = sample[0], sample[1]
                img = tf.keras.preprocessing.image.load_img(filename)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_bytes = tf.io.serialize_tensor(img_array)
                image_shape = img_array.shape
                example = data_utils.serialize_example(img_bytes, class_id, image_shape)
                
                tfrecord_writer.write(example)
                
            except IOError as e:
                print("Could not read:", file_names[i])
                print("Error", e)
                print("Skip~\n")
                
    sys.stdout.write('\n')
    sys.stdout.flush()

    
def clean_up_temporary_files(local_filepath, uncompressed_dir, split_data_dir):
    """Removes temporary files used to create the dataset.
    Args:
        local_filepath: The local path where the downloaded is stored.
        uncompressed_dir: The directory where the temporary files are stored.
    Returns:
        None
    """
    tf.io.gfile.remove(local_filepath)
    tf.io.gfile.rmtree(uncompressed_dir)
    tf.io.gfile.rmtree(split_data_dir)

    
def run(dataset_dir, split_ratio = {'train':0.7, 'dev': 0.15, 'test': 0.15}):
    """ Runs the download and conversion operation.
    Args:
        dataset_dir: The dataset directory where the dataset is stored.
        split_rate: The dictionary represents split name and ratio
    """
    # Create dataset folder for convertion if necessary
    if not tf.io.gfile.exists(dataset_dir):
        tf.io.gfile.mkdir(dataset_dir)

    # Examine whether tfrecord exists
    tfrecord_files = zip(split_ratio.keys(),
                         map(lambda x: get_tfrecord_filename(dataset_dir, x), split_ratio.keys()))
    tfrecord_files = dict(tfrecord_files)
    if all(map(lambda x: tf.io.gfile.exists(x), tfrecord_files.values())):
        print('TFRecords files already exist. Exiting without re-creating them.')
        return

    # Download and decompress to the designated location
    downloaded_filepath = os.path.join(dataset_dir, _DATA_NAME)
    uncompressed_dir = os.path.join(dataset_dir, 'tmp')
    split_data_dir = os.path.join(dataset_dir, 'split')
    data_utils.download_and_uncompress(_DATA_URL, downloaded_filepath, uncompressed_dir)
    
    # Make saparation for each class into train, valid and test
    saparate_dataset(uncompressed_dir, split_data_dir, ratio=split_ratio)
    
    # Get mapping for class names and ids and gather all samples from directory
    class_names = get_classnames(uncompressed_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    
    # Transform to TFRecords
    for split_type in tfrecord_files:
        sample_path = os.path.join(split_data_dir, split_type)
        samples = get_samples(sample_path, class_names_to_ids)
        convert_dataset(samples, tfrecord_files[split_type])
  
    # Write the mapping of ids to class name
    ids_to_class_names = dict(zip(range(len(class_names)), class_names))
    data_utils.write_label_file(ids_to_class_names, dataset_dir)

    # clean the intermediate files
    clean_up_temporary_files(downloaded_filepath, uncompressed_dir, split_data_dir)
    
    # All Done
    print('\nFinished converting the dataset!')
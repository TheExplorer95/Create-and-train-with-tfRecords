import os
from math import log, floor
from pathlib import Path
import argparse

import tensorflow as tf

from data_pipeline import load_image, preprocess_image_tfRecord, preprocess_label, preprocess_dataset, get_data_generator_single
from optional import count_files_by_extension, create_dirs, str2bool
from configuration import constants


# Command line interface -----------------------
def get_command_line_args():
    """Parses command line arguments that are passed to the script"""

    parser = argparse.ArgumentParser(description='Define script params')

    # paths
    parser.add_argument("-o", '--output_dir', type=str, default='tfrecord_data/')

    # script
    parser.add_argument('-s', '--imgs_per_shard', type=int, default=1000)
    parser.add_argument('-dn', '--dataset_name', type=str, default='imagenet')
    parser.add_argument("-dt", '--dataset_type', type=str, default='val')  # val, test, train

    # data preprocessing
    parser.add_argument('-p', '--data_preprocessing', type=str2bool, default=True)
    parser.add_argument('-a', '--architecture', type=str, default='VGG16')  # only necessary if preprocessing is used

    return vars(parser.parse_args())


# functions to create tfRecords ------------------------------
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""

    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def map_serializaton(img, label):
    """Wrapper function that enables serialize datapoint to work with the
    tf.dataset.Dataset.map function."""

    def serialize_datapoint(img, label):
        """Returns a serialized version of the img and label provided."""
        feature = {'img_raw': _bytes_feature(tf.io.serialize_tensor(img)),
                   'label': _int64_feature(label)}

        features = tf.train.Features(feature=feature)
        feature_proto = tf.train.Example(features=features)
        serialized_data = feature_proto.SerializeToString()

        return serialized_data

    serialized_data = tf.py_function(serialize_datapoint,
                                     (img, label),
                                     tf.string)

    return serialized_data


def get_dataset_path(dataset_type, dataset):
    """
    Parses and checks the dataset path. If the path provided to the CLI is eihter
    val_imgnet or train_imgnet, the path is set to the one on the IKW computing
    cluster.
    """
    from configuration import constants
    dataset_config = constants.DATASETS[dataset]

    if dataset_type == 'val':
        data_dir = Path(dataset_config['val_path'])
    elif dataset_type == 'test':
        data_dir = Path(dataset_config['test_path'])
    elif dataset_type == 'train':
        data_dir = Path(dataset_config['train_path'])

    if not data_dir.is_dir():
        raise ValueError(f'The provided data set directory {path} is no valid path.')

    return data_dir

def create_tfRecord_from_dataset(ds, n_datapoints):
    """
    Based on a tf.dataset.Dataset object, whichs content is the serialized dataset,
    tfRecord files are created.

    n_datapoints: The amount of datapoints within the dataset
    """

    n_shards = (n_datapoints//args['imgs_per_shard']) + 1
    if n_datapoints%args['imgs_per_shard'] == 0:
        n_shards -= 1

    # create shards
    shard_power_10 = floor(log(n_shards, 10)) + 1  # used for creating file names
    shard_count = 0
    file_count = 0

    for serialized_data in ds:
        # instantiate new writer
        if file_count % args['imgs_per_shard'] == 0:
            if shard_count != 0:
                writer.close()

            print(f'Currently at file {file_count:0>2d} of {n_datapoints}, generating shard {shard_count} of {n_shards-1}.')

            tmp_shard_filename = f'{shard_count:0>{shard_power_10}d}_{n_shards-1}-data.tfrecords'
            tmp_shard_filepath = out_dir.joinpath(tmp_shard_filename)
            writer = tf.io.TFRecordWriter(str(tmp_shard_filepath))

            shard_count += 1

        # write serialized data to the shard
        writer.write(serialized_data.numpy())
        file_count += 1

    writer.close()

# functions to extract tfRecords --------------
def _parse_function(example_proto):
    """
    Used as map function for tf.dataset.Dataset to extract img and label data
    from the created tfRecords files.
    """

    feature_description = {'img_raw': tf.io.FixedLenFeature([], tf.string),
                           'label': tf.io.FixedLenFeature([], tf.int64)}

    feature = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.io.parse_tensor(feature['img_raw'], out_type=tf.float32)

    return img, feature['label']

def create_dataset_from_tfRecords(filepaths: list, n_classes, batch_size, interleave_cycle=4, shuffle_buffer=8192):
    """
    Creates a tf.dataset.Dataset object from the passed list of tfRecords files.
    As the labels were not one-hot encoded before saving, this is also taken
    care of.

    filepaths: list of relative or absolute paths to each tfRecord file
    n_classes: number of classes
    """

    # 1. create dataset from the filepaths and shuffle them
    fn_dataset = tf.data.Dataset.from_tensor_slices(filepaths).shuffle(len(filepaths))

    # 2. create dataset of the serialized data and shuffle again
    serialized_dataset = fn_dataset.interleave(tf.data.TFRecordDataset,
                                               cycle_length=interleave_cycle,
                                               num_parallel_calls=tf.data.AUTOTUNE)
    serialized_dataset = serialized_dataset.shuffle(buffer_size=shuffle_buffer)

    # 3. Parse and preprocess the serialized dataset
    parsed_dataset = serialized_dataset.map(_parse_function)
    parsed_dataset = parsed_dataset.map(lambda img, label: (img, preprocess_label(label, n_classes)))
    parsed_dataset = parsed_dataset.batch(6).prefetch(6)

    return parsed_dataset

if __name__ == '__main__':
    args = get_command_line_args()

    # 1. check directories
    data_dir = get_dataset_path(dataset_type=args['dataset_type'],
                                dataset=args['dataset_name'])

    # define the output dir
    out_dir = Path(args['output_dir']).joinpath(args['dataset_type'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. create generator
    gen_output_signature = (tf.TensorSpec([None, None, 3], dtype=tf.float32),
                            tf.TensorSpec((), dtype=tf.int32))

    data_gen = get_data_generator_single(data_directory=data_dir,
                                         n_classes=constants.DATASETS[args['dataset_name']]['n_classes'])

    ds = tf.data.Dataset.from_generator(generator=data_gen,
                                        output_signature=gen_output_signature)

    # 3. preprocess images
    if args['data_preprocessing']:
        ds = ds.map(lambda img, label: (preprocess_image_tfRecord(img,
                                                                  dataset=args['dataset_name'],
                                                                  architecture=args['architecture']),
                                        label),
                    num_parallel_calls=tf.data.AUTOTUNE)

    # 4. serialize data
    ds = ds.map(lambda img, label: map_serializaton(img, label))

    # 5. save serialized data to tfRecords
    create_tfRecord_from_dataset(ds=ds,
                                 n_datapoints=count_files_by_extension(data_dir, 'JPEG'))

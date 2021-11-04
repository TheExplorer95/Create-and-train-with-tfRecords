import tensorflow as tf
import random
import os
from configuration import constants

def load_image(img_path):
    """
    Given a path to a JPEG the image is loaded as a tf.Tensor.
    """
    raw = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(raw, channels=3)

    return image

@tf.function
def preprocess_image_tfRecord(image, dataset, architecture):
    """
    Based on the given preprocessing configuration, the given image is preprocessed.
    Intended to be used as a map function for a tf.dataset.Dataset.
    IMPORTANT: Do not use data augmentation before saving the data on disk.
    """

    train_const = constants.TRAIN_CONFIGS[architecture]
    dataset_const = constants.DATASETS[dataset]

    assert train_const['dataset'] == dataset, f'The provided train_config was created for the {train_const["dataset"]} dataset and cannot be used for preprocessing of the {dataset} dataset. Adapt your train config inside the configuration.py file.'

    if train_const['preprocessing']['mean_subtraction']:
        image -= dataset_const['mean_train']

    if train_const['preprocessing']['standardization']:
        image -= dataset_const['mean_train']
        image /= dataset_const['std_train']

    if train_const['preprocessing']['convert_to_BGR']:
        image = image[..., ::-1]

    # rescales the imput between [0, 1]
    if train_const['preprocessing']['scale_01']:
        image /= 255

    # rescales the imput between [-1, 1]
    if train_const['preprocessing']['scale_11']:
        image /= 127.5
        image -= 1.

    # images are isotropically rescales to size_smallest_img_side
    if train_const['preprocessing']['rescale_dimensions'] == 'isotropically':
        size_smallest_img_side = train_const['preprocessing']['size_smallest_img_side']
        image_shape = tf.cast(tf.shape(image), dtype=tf.float64)

        min_img_size = tf.math.reduce_min(image_shape[:-1])
        difference = tf.cast(min_img_size - size_smallest_img_side, dtype=tf.bool)

        if difference:
            rescale_factor = size_smallest_img_side / min_img_size
            rescaled_height = image_shape[0] * rescale_factor
            rescaled_width = image_shape[1] * rescale_factor

            desired_img_shape = (rescaled_height, rescaled_width)
            image = tf.image.resize(image, desired_img_shape)

    # images are rescaled but aspect ratio distorted
    elif train_const['preprocessing']['rescale_dimensions'] == 'distorted':
        resized_shape = train_const['input_shape'] if len(train_const['input_shape']) <= 2 else train_const['input_shape'][:2]
        image = tf.image.resize(image, resized_shape)

    return image


@tf.function
def preprocess_label(labels, classes):
    """One hot encoding for labels"""

    one_hot_label = tf.one_hot(labels, classes)

    return one_hot_label

def preprocess_dataset(dataset, augment_data, shuffle_buffer=None, batch_size=None, prefetch_buffer=None):
    """Handles preprocessing of a tf.Dataset"""

    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    if augment_data:
        data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)])

        dataset = dataset.map(lambda img, label: (data_augmentation(img, training=True), label),
                              num_parallel_calls=tf.data.AUTOTUNE)

    if prefetch_buffer is not None:
        dataset = dataset.prefetch(prefetch_buffer)

    return dataset

def get_data_generator_single(data_directory, n_classes):
    """
    Wrapper function that loads all filepaths for a given datset and returns a generator.
    Instead of loading the image into RAM, only the paths are loaded before training
    and during forward pass the generator loads the actual image. Does not
    produce batches of images, hence the name single.

    File Structure:
    dataset_name
      -class_name
        -images
    """
    class_paths = sorted([class_dir.path for class_dir in os.scandir(data_directory) if class_dir.is_dir()])
    label_to_id = {class_label: id for id, class_label in enumerate([path.split('/')[-1] for path in class_paths])}

    img_paths = []
    labels = []
    for class_path in class_paths:
        tmp_paths = [img.path for img in os.scandir(class_path)]
        img_paths.extend(tmp_paths)
        labels.extend([label_to_id[f'{class_path.split("/")[-1]}']] * len(tmp_paths))

    data = [(img_path, label) for img_path, label in zip(img_paths, labels)]
    random.shuffle(data)

    def data_generator():
        for img_path, label in zip(img_paths, labels):
            img = load_image(img_path)
            label = tf.constant(label, dtype=tf.int32)

            yield img, label

    return data_generator


# old but gold ==============

@tf.function
def preprocess_image(image, dataset, convert_to_BGR, scale_01, scale_11,
                     mean_subtraction, standardization,
                     size_smallest_img_side=None, network_input_shape=None, flip_l_r=False):
    """
    Based on given parameters, the given image is preprocesses. If used
    for normalization or standardization, the dataset name must be provided and
    it's standard deviation and mean saved within the configuration file.
    Intended to be used as a map function for a tf.dataset.Dataset.
    IMPORTANT: Do not use data augmentation before saving the data on disk.
    """

    if mean_subtraction or standardization:
        ds_train_mean = constants.DATASETS[dataset]['mean_train']
        image -= ds_train_mean

    if standardization:
        ds_train_std = constants.DATASETS[dataset]['std_train']
        image /= ds_train_std

    if convert_to_BGR:
        image = image[..., ::-1]

    if scale_01:
        image /= 255

    if scale_11:
        image /= 127.5
        image -= 1.

    # VGG16 data preparation
    # 1. images are isotropically rescales to size_smallest_img_side
    if size_smallest_img_side is not None:
        # downsample image to desired shape (does it like the VGG16 paper)
        image_shape = tf.cast(tf.shape(image), dtype=tf.float64)

        min_img_size = tf.math.reduce_min(image_shape[:-1])
        difference = tf.cast(min_img_size - size_smallest_img_side, dtype=tf.bool)

        if difference:
            rescale_factor = size_smallest_img_side / min_img_size
            rescaled_height = image_shape[0] * rescale_factor
            rescaled_width = image_shape[1] * rescale_factor

            desired_img_shape = (rescaled_height, rescaled_width)
            image = tf.image.resize(image, desired_img_shape)

    # 2. randomly takes a crop of the rescaled image
    if network_input_shape is not None:
        image = tf.image.random_crop(image, network_input_shape)

    # 3. images are randomly vertically flipped
    if flip_l_r:
        image = tf.image.random_flip_left_right(image)

    return image

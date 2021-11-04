import types
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

"""
1. Example of a data set constants dict:
    n_classes: the amount of classes within the dataset
    mean_train: mean of the training datset for each channel
    std_train: standard deviation of the training datset for each channel
    val_path: the path to the validation dataset
    test_path: the path to the test dataset
    train_path: the path to the train dataset

ecoset_constants = {'n_classes': tf.constant(..., dtype=tf.int32),
                    'mean_train': tf.constant(..., dtype=tf.float32),
                    'std_train': tf.constant(..., dtype=tf.float32)}
                    'val_path': '/net/projects/data/ImageNet/ILSVRC2012/val2',
                    'test_path': None,
                    'train_path': '/net/projects/data/ImageNet/ILSVRC2012/train'}

2. Example of a model train config:
2.1 for tfRecords creation
    dataset: the dataset to be analyzed (must be one that is implementet as constants.DATASETS)
    input_shape: size of the images that are fed into the model
    preprocessing: dict that specifies the preprocessing procedures based on the
                   preprocess_image_tfRecord function (found within data_pipeline)
        convert_to_BGR: used to change the color channel order of an image
        mean_subtraction: subtracts the mean of the train dataset from each image
        scale_01: scales each image within the range of [0, 1]
        scale_11: scales each image within the range of [-1, 1]
        standardization: standardizes the image (mean subtraction and standard
                         deviation division)
        rescale_dimensions: one of [isotropically, distorted]; defines the method
                            which is used to rescale images to the same size.
                                - Isotropically takes the smallest side of an image
                                and rescales it to the size of size_smallest_img_side,
                                while taking into account that the other side is
                                scaled by the same factor.
                                - Distorted scales the image to the defined
                                input_shape for the corresponding model, while not
                                preserving the aspect ratio
        size_smallest_img_side: smallest side of an isotropically-rescaled training image

VGG16_train_config = {'dataset': 'imagenet',
                      'min_img_shape': 256,
                      'input_shape': (224, 224, 3),
                      'preprocessing': {'convert_to_BGR': True,
                                        'mean_subtraction': True,
                                        'scale_01': False,
                                        'scale_11': False,
                                        'standardization': False,
                                        'rescale_dimensions': 'isotropically',
                                        'size_smallest_img_side': 256}}

2.2 for training od the model
    batch_size: amount of datapoints per batch
        optimizer: the optimizer used during training (needs to be initialized)

    to be continued

"""



# Storeage for constants -----------------
constants = types.SimpleNamespace()

# Data input pipeline
imageNet_constants = {'n_classes': tf.constant(1000, dtype=tf.int32),
                      'mean_train': tf.constant([121.6189, 116.635, 102.7882], dtype=tf.float32),
                      'std_train': tf.constant([58.395, 57.12, 57.375], dtype=tf.float32),
                      # 'mean_val': tf.constant([119.5536, 114.7759, 101.07697], dtype=tf.float32),
                      'val_path': '/net/projects/data/ImageNet/ILSVRC2012/val2',
                      'train_path': '/net/projects/data/ImageNet/ILSVRC2012/train'}

constants.DATASETS = {'imagenet': imageNet_constants}

# Preprocessing pipeline
VGG16_train_config = {'dataset': 'imagenet',
                      'input_shape': (224, 224, 3),
                      'preprocessing': {'convert_to_BGR': True,
                                        'mean_subtraction': True,
                                        'scale_01': False,
                                        'scale_11': False,
                                        'standardization': False,
                                        'rescale_dimensions': 'distorted',
                                        'size_smallest_img_side': 256},
                      'batch_size': 64,
                      'optimizer': SGD(learning_rate=1e-3, momentum=0.9)}

VGG16_regularized_train_config = {'dataset': 'imagenet',
                                  'min_img_shape': 256,
                                  'input_shape': (224, 224, 3),
                                  'preprocessing': {'convert_to_BGR': True,
                                                    'mean_subtraction': True,
                                                    'scale_01': False,
                                                    'scale_11': False,
                                                    'standardization': False,
                                                    'rescale_dimensions': 'isotropically',
                                                    'size_smallest_img_side': 256},
                                  'batch_size': 64,
                                  'optimizer': SGD(learning_rate=1e-4, momentum=0.9)}

EfficientNet_train_config = {'dataset': 'imagenet',
                             'input_shape': (260, 260, 3),
                             'preprocessing': {'convert_to_BGR': False,
                                                        'mean_subtraction': False,
                                                        'scale_01': False,
                                                        'scale_11': False,
                                                        'standardization': False,
                                                        'rescale_dimensions': 'distorted',
                                                        'size_smallest_img_side': None},
                             'batch_size': 32,
                             'optimizer': Adam(learning_rate=1e-3)}

InceptionResnet50_train_config = {'dataset': 'imagenet',
                                  'input_shape': (299, 299, 3),
                                  'preprocessing': {'convert_to_BGR': False,
                                                             'mean_subtraction': False,
                                                             'scale_01': False,
                                                             'scale_11': True,
                                                             'standardization': False,
                                                             'rescale_dimensions': 'distorted',
                                                             'size_smallest_img_side': None},
                                  'batch_size': 32,
                                  'optimizer': Adam(learning_rate=1e-3)}

InceptionV3_train_config = {'dataset': 'imagenet',
                            'input_shape': (299, 299, 3),
                            'preprocessing': {'convert_to_BGR': False,
                                              'mean_subtraction': False,
                                              'scale_01': False,
                                              'scale_11': True,
                                              'standardization': False,
                                              'rescale_dimensions': 'distorted',
                                              'size_smallest_img_side': None},
                            'batch_size': 64,
                            'optimizer': Adam(learning_rate=1e-3)}

Xception_train_config = {'dataset': 'imagenet',
                         'input_shape': (299, 299, 3),
                         'preprocessing': {'convert_to_BGR': False,
                                           'mean_subtraction': False,
                                           'scale_01': False,
                                           'scale_11': True,
                                           'standardization': False,
                                           'rescale_dimensions': 'distorted',
                                           'size_smallest_img_side': None},
                         'batch_size': 64,
                         'optimizer': Adam(learning_rate=1e-3)}

constants.TRAIN_CONFIGS = {'VGG16': VGG16_train_config,
                           'VGG16_regularized': VGG16_regularized_train_config,
                           'EfficientNet': EfficientNet_train_config,
                           'InceptionResnet50': InceptionResnet50_train_config,
                           'InceptionV3': InceptionV3_train_config,
                           'Xception': Xception_train_config}

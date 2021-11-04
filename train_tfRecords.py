import os  #; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from optional import activate_memory_growth; activate_memory_growth()
from tfRecords import create_dataset_from_tfRecords
from configuration import constants
from optional import get_file_paths_by_extension

train_config = constants.TRAIN_CONFIGS['VGG16']
dataset_config = constants.DATASETS['imagenet']

# 1. Dataset =========================
print(f'[INFO] - Initializing the input pipeline for the dataset.')

val_filepaths = get_file_paths_by_extension(dir='tfrecord_data/val',
                                            extension='tfrecords')
train_filepaths = get_file_paths_by_extension(dir='tfrecord_data/train',
                                              extension='tfrecords')

val_ds = create_dataset_from_tfRecords(filepaths=val_filepaths,
                                       n_classes=dataset_config['n_classes'],
                                       batch_size=train_config['batch_size'])

train_ds = create_dataset_from_tfRecords(filepaths=train_filepaths,
                                         n_classes=dataset_config['n_classes'],
                                         batch_size=train_config['batch_size'])

# 2. Model ============================
print(f'[INFO] - Initializing and training the model.')
model = tf.keras.applications.vgg16.VGG16(weights=None,
                                          include_top=True,
                                          input_shape=train_config['input_shape'])

train_optimizer = tf.keras.optimizers.Adam(learning_rate=0.000_1)

model.compile(optimizer=train_optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

train_history = model.fit(x=train_ds,
                          validation_data=val_ds)

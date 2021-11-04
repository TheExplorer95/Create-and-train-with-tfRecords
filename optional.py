import tensorflow as tf
import os
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

def count_files_by_extension(dir, extension):
    wildcard_extension = f'*.{extension}'
    counter = 0

    for path, subdirs, files in os.walk(dir):
        for name in files:
            if fnmatch(name, wildcard_extension):
                counter += 1

    return counter


def get_file_paths_by_extension(dir, extension):
    wildcard_extension = f'*.{extension}'

    file_paths = []
    for path, subdirs, files in os.walk(dir):
        for file in sorted(files):
            if fnmatch(file, wildcard_extension):
                file_paths.append(os.path.join(path, file))

    return file_paths


def activate_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) == 1:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print('Tensorflow GPU memory growth: activated')
        except Exception as e:
            print('Tensorflow GPU memory growth: deactivated')
            print(str(e))


def create_dirs(dirs):
    if isinstance(dirs, list):
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
    elif isinstance(dirs, str):
        os.makedirs(dirs, exist_ok=True)


def get_time_str():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {value} with type {type(value)}')


class Print_Metrics_At_EpochEnd(tf.keras.callbacks.Callback):
    """
    Callback that prints the metrics only at epoch end.
    """

    def __init__(self, print_fct):
        super(Print_Metrics_At_EpochEnd, self).__init__()
        self.print_fct = print_fct
        # self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        self.print_fct(f'[EPOCH - {epoch}]')
        for metric, value in logs.items():
            self.print_fct(f'{metric}: {value}')

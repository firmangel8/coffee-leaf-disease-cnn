import os
import math
import shutil
import numpy as np


def init_data_lab_storage(class_name):
    os.makedirs('data_lab/train/' + class_name, exist_ok=True)
    os.makedirs('data_lab/validation/' + class_name, exist_ok=True)
    os.makedirs('data_lab/test/' + class_name, exist_ok=True)


def copy_batch_file(sources, destination, portion='auto'):
    files_sources = os.listdir(sources)
    quota = 0
    total_files = len(files_sources)
    d1 = int(math.ceil(total_files * 0.8 * 0.8))
    d2 = int(math.ceil(total_files * 0.8 * 0.2))
    if portion == 'auto':
        quota = total_files - (d1 + d2)
    else:
        quota = int(math.ceil(len(files_sources) * portion))
    for file in files_sources[:quota]:
        shutil.copy(sources + file, destination + file)


def distribute_based_on_class(class_name):
    # batch move based on class name
    init_data_lab_storage(class_name)
    copy_batch_file('./data_class_generated_group/' + class_name + '/',
                    './data_lab/train/' + class_name + '/', (0.8 * 0.8))
    copy_batch_file('./data_class_generated_group/' + class_name + '/',
                    './data_lab/validation/' + class_name + '/', 0.2 * 0.8)
    copy_batch_file('./data_class_generated_group/' + class_name + '/',
                    './data_lab/test/' + class_name + '/', 'auto')


my_class = ['miner', 'nominer', 'rust', 'norust', 'phoma', 'nophoma']
for x in my_class:
    distribute_based_on_class(x)

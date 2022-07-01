import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import tensorflow as tf

from pathlib import Path
Path("data_class_generated/").mkdir(parents=True, exist_ok=True)


import os

from matplotlib import pyplot
from matplotlib.image import imread

from distutils.dir_util import copy_tree
from shutil import copy

from tensorflow.keras import layers, models

from matplotlib import pyplot as plt

import keras as keras

from keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import tensorflow as tf
import tensorflow.keras.metrics as tfm


te = pd.read_csv('test_classes.csv')
print("All ", te.count())
print("Miner ", te[te.miner > 0].count())
print("Rust ", te[te.rust > 0].count())
print("Phoma ", te[te.phoma > 0].count())
print("None ", te[(te.miner == 0) & (te.rust == 0) & (te.phoma == 0)].count())

te = pd.read_csv('train_classes.csv')
print("All ", te.count())
print("Miner ", te[te.miner > 0].count())
print("Rust ", te[te.rust > 0].count())
print("Phoma ", te[te.phoma > 0].count())
print("None ", te[(te.miner == 0) & (te.rust == 0) & (te.phoma == 0)].count())


# Preparation for binary classification (miner)
os.makedirs('data_class_generated_group/miner', exist_ok=True)
os.makedirs('data_class_generated_group/nominer', exist_ok=True)
os.makedirs('data_class_generated_group/miner', exist_ok=True)
os.makedirs('data_class_generated_group/nominer', exist_ok=True)
trl = pd.read_csv('train_classes.csv')
for index, row in trl.iterrows():
    if(row['miner'] == 1):
        copy('coffee-leaf-diseases/train/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/miner/miner_' + str(row['id']) + '.jpg')
    else:
        copy('coffee-leaf-diseases/train/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/nominer/nominer_' + str(row['id']) + '.jpg')

tel = pd.read_csv('test_classes.csv')
for index, row in tel.iterrows():
    if(row['miner'] == 1):
        copy('coffee-leaf-diseases/test/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/miner/miner_' + str(row['id']) + '.jpg')
    else:
        copy('coffee-leaf-diseases/test/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/nominer/nominer_' + str(row['id']) + '.jpg')

# Preparation for binary classification (rust)
os.makedirs('data_class_generated_group/rust', exist_ok=True)
os.makedirs('data_class_generated_group/norust', exist_ok=True)
os.makedirs('data_class_generated_group/rust', exist_ok=True)
os.makedirs('data_class_generated_group/norust', exist_ok=True)
trl = pd.read_csv('train_classes.csv')
for index, row in trl.iterrows():
    if(row['rust'] == 1):
        copy('coffee-leaf-diseases/train/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/rust/rust_' + str(row['id']) + '.jpg')
    else:
        copy('coffee-leaf-diseases/train/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/norust/norust_' + str(row['id']) + '.jpg')

tel = pd.read_csv('test_classes.csv')
for index, row in tel.iterrows():
    if(row['rust'] == 1):
        copy('coffee-leaf-diseases/test/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/rust/rust_' + str(row['id']) + '.jpg')
    else:
        copy('coffee-leaf-diseases/test/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/norust/norust_' + str(row['id']) + '.jpg')

# Preparation for binary classification (phoma)
os.makedirs('data_class_generated_group/phoma', exist_ok=True)
os.makedirs('data_class_generated_group/nophoma', exist_ok=True)
os.makedirs('data_class_generated_group/phoma', exist_ok=True)
os.makedirs('data_class_generated_group/nophoma', exist_ok=True)
trl = pd.read_csv('train_classes.csv')
for index, row in trl.iterrows():
    if(row['phoma'] == 1):
        copy('coffee-leaf-diseases/train/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/phoma/phoma_' + str(row['id']) + '.jpg')
    else:
        copy('coffee-leaf-diseases/train/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/nophoma/nophoma_' + str(row['id']) + '.jpg')

tel = pd.read_csv('test_classes.csv')
for index, row in tel.iterrows():
    if(row['phoma'] == 1):
        copy('coffee-leaf-diseases/test/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/phoma/phoma_' + str(row['id']) + '.jpg')
    else:
        copy('coffee-leaf-diseases/test/images/' +
             str(row['id']) + '.jpg', 'data_class_generated_group/nophoma/nophoma_' + str(row['id']) + '.jpg')
print('generated success...')

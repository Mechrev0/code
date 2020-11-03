import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from .dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y,classes = load_data()

index = 0
plt.imshow(train_x_orig[index])
print(f'y = {str(train_y[0, index])}. It is a {classes[train_y[0, index]].decode("utf-8")} picture.')

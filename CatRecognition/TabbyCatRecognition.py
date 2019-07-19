#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object Classification(such as a Tabby Cat image) using pretrained Inception v3

1. The test image is located in ./images/cnn/test_image.png
2. Download the whole folder and run the .py file directly in your workspace
3. Downloading the pretrained Inception v3 may take a while (size of 
   inception_v3.ckpt: 103.8 MB)

"""


###==========Preparation==========

from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# reproductivity
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Save the figures in case
PROJECT_ROOT_DIR = "."
SUBJECT_ID = "cnn"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", SUBJECT_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")


###==========Classifying images using Inception v3==========

'''
(1)Download some images of various animals. Load them in Python using the
matplotlib.image.mpimg.imread() function or the scipy.misc.imread() function
(2)Resize and/or crop them to 299 × 299 pixels, and ensure that they 
have just three channels (RGB), with no transparency channel. 
(3)The images that the Inception model was trained on were preprocessed so that
 their values range from -1.0 to 1.0, so ensure that your images do too.
'''

width = 299
height = 299
channels = 3

import matplotlib.image as mpimg
test_image = mpimg.imread(os.path.join("images","cnn","test_image2.png"))[:, :, :channels]
plt.imshow(test_image)
plt.axis("off")
plt.show()

test_image = 2 * test_image - 1

# Download the latest pretrained Inception v3 model
import sys                                                                 
import tarfile
from six.moves import urllib

TF_MODELS_URL = "http://download.tensorflow.org/models"
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()

def fetch_pretrained_inception_v3(url=INCEPTION_V3_URL, path=INCEPTION_PATH):
    if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "inception_v3.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    inception_tgz = tarfile.open(tgz_path)
    inception_tgz.extractall(path=path)
    inception_tgz.close()
    os.remove(tgz_path)

fetch_pretrained_inception_v3()

import re

CLASS_NAME_REGEX = re.compile(r"^n\d+\s+(.*)\s*$", re.M | re.U)

def load_class_names():
    with open(os.path.join("datasets", "inception", "imagenet_class_names.txt"), "rb") as f:
        content = f.read().decode("utf-8")
        return CLASS_NAME_REGEX.findall(content)

class_names = ["background"] + load_class_names()                           

# Create the Inception v3 model by calling the inception_v3() function
from tensorflow.contrib.slim.nets import inception                          
import tensorflow.contrib.slim as slim

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
with slim.arg_scope(inception.inception_v3_arg_scope()):                    
    logits, end_points = inception.inception_v3(
        X, num_classes=1001, is_training=False)
predictions = end_points["Predictions"]                                    
saver = tf.train.Saver()

# Run the model to classify the images you prepared. Display the top five
# predictions for each image, as well as the estimated probability. 
X_test = test_image.reshape(-1, height, width, channels)

with tf.Session() as sess:
    saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
    predictions_val = predictions.eval(feed_dict={X: X_test})

most_likely_class_index = np.argmax(predictions_val[0])
most_likely_class_index
class_names[most_likely_class_index]

top_5 = np.argpartition(predictions_val[0], -5)[-5:]
top_5 = reversed(top_5[np.argsort(predictions_val[0][top_5])])
for i in top_5:
    print("{0}: {1:.2f}%".format(class_names[i], 100 * predictions_val[0][i]))

'''
Result:
    tabby, tabby cat: 85.80%
    tiger cat: 10.30%
    Egyptian cat: 0.48%
    lynx, catamount: 0.22%
    Persian cat: 0.22%
'''








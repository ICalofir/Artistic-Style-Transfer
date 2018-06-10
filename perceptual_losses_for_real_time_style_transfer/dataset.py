import json
import os
import random
import tensorflow as tf
import numpy as np
from io import BytesIO

class Dataset:
  def __init__(self,
      data_path='perceptual_losses_for_real_time_style_transfer/dataset'):
    self._x_train = None
    self._x_val = None
    self.data_path = data_path

    self.train_batch_idx = 0

  def get_train_batch(self, n_batch=None):
    if self._x_train is None: self._get_dataset()

    if n_batch is None:
      return self._x_train

    batch_end = False

    x_batch = [self.data_path + '/train_imgs/' + self._x_train[i]
               for i in list(range(
                 self.train_batch_idx,
                 min(self.train_batch_idx + n_batch,
                     len(self._x_train))))]

    if self.train_batch_idx + n_batch < len(self._x_train):
      self.train_batch_idx = self.train_batch_idx + n_batch
    else:
      batch_end = True

    return x_batch, batch_end

  def get_val_batch(self, n_batch=None):
    if self._x_val is None:
      self._get_dataset()

    if n_batch is None:
      return self._x_val

    ind = np.random.randint(len(self._x_val), size=n_batch)
    x_batch = [self.data_path + '/val_imgs/' + self._x_val[i] for i in ind]

    return x_batch

  def _get_dataset(self):
    s = tf.InteractiveSession()
    val_tf = tf.read_file(self.data_path + '/anno/val.txt')
    train_tf = tf.read_file(self.data_path + '/anno/train.txt')

    val_bytes = val_tf.eval()
    train_bytes = train_tf.eval()
    s.close()

    val_imgs = BytesIO(val_bytes).read().decode('utf-8').split('\n')
    train_imgs = BytesIO(train_bytes).read().decode('utf-8').split('\n')

    self._x_val = []
    for line in val_imgs:
      img_line = line.rstrip() # remove newline
      self._x_val.append(img_line)
    random.shuffle(self._x_val)

    self._x_train = []
    for line in train_imgs:
      img_line = line.rstrip() # remove newline
      self._x_train.append(img_line)
    random.shuffle(self._x_train)

    # self._x_val = []
    # if os.path.exists(self.data_path + '/anno/val.txt'):
      # with open(self.data_path + '/anno/val.txt') as f:
        # for line in f:
          # img_line = line.rstrip() # remove newline
          # self._x_val.append(img_line)

        # random.shuffle(self._x_val)

    # self._x_train = []
    # if os.path.exists(self.data_path + '/anno/train.txt'):
      # with open(self.data_path + '/anno/train.txt') as f:
        # for line in f:
          # img_line = line.rstrip() # remove newline
          # self._x_train.append(img_line)

        # random.shuffle(self._x_train)

  @staticmethod
  def build_dataset_coco_microsoft(
      data_path='perceptual_losses_for_real_time_style_transfer/dataset/anno/train_original.json',
      output_file='perceptual_losses_for_real_time_style_transfer/dataset/anno/train.txt'):
    with open(data_path) as f:
      data = json.load(f)

    with open(output_file, 'w') as f:
      for image in data['images']:
        f.write(image['file_name'] + '\n')

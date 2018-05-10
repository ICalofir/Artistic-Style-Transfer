import os
import random
import json

class Dataset:
  def __init__(self,
      data_path='perceptual_losses_for_real_time_style_transfer/dataset'):
    self._x_train = None
    self._x_val = None
    self._x_test = None
    self.data_path = data_path

    self.train_batch_idx = 0
    self.val_batch_idx = 0
    self.test_batch_idx = 0

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

    batch_end = False

    x_batch = [self.data_path + '/val_imgs/' + self._x_val[i]
               for i in list(range(
                 self.val_batch_idx,
                 min(self.val_batch_idx + n_batch,
                     len(self._x_val))))]

    if self.val_batch_idx + n_batch < len(self._x_val):
      self.val_batch_idx = self.val_batch_idx + n_batch
    else:
      batch_end = True

    return x_batch, batch_end

  def get_test_batch(self, n_batch=None):
    if self._x_test is None:
      self._get_dataset()

    if n_batch is None:
      return self._x_test

    batch_end = False

    x_batch = [self.data_path + '/test_imgs/' + self._x_test[i]
               for i in list(range(
                 self.test_batch_idx,
                 min(self.test_batch_idx + n_batch,
                     len(self._x_test))))]

    if self.test_batch_idx + n_batch < len(self._x_test):
      self.test_batch_idx = self.test_batch_idx + n_batch
    else:
      batch_end = True

    return x_batch, batch_end

  def _get_dataset(self):
    self._x_test = []
    if os.path.exists(self.data_path + '/anno/test.txt'):
      with open(self.data_path + '/anno/test.txt') as f:
        for line in f:
          img_line = line.rstrip() # remove newline
          self._x_test.append(img_line)

        random.shuffle(self._x_test)

    self._x_val = []
    if os.path.exists(self.data_path + '/anno/val.txt'):
      with open(self.data_path + '/anno/val.txt') as f:
        for line in f:
          img_line = line.rstrip() # remove newline
          self._x_val.append(img_line)

        random.shuffle(self._x_val)

    self._x_train = []
    if os.path.exists(self.data_path + '/anno/train.txt'):
      with open(self.data_path + '/anno/train.txt') as f:
        for line in f:
          img_line = line.rstrip() # remove newline
          self._x_train.append(img_line)

        random.shuffle(self._x_train)

  @staticmethod
  def build_dataset_coco_microsoft(
      data_path='perceptual_losses_for_real_time_style_transfer/dataset/anno/train_original.json',
      output_file='perceptual_losses_for_real_time_style_transfer/dataset/anno/train.txt'):
    with open(data_path) as f:
      data = json.load(f)

    with open(output_file, 'w') as f:
      for image in data['images']:
        f.write(image['file_name'] + '\n')

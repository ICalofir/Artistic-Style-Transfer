import cv2
import numpy as np

try:
  from perceptual_losses_for_real_time_style_transfer.dataset import Dataset
except ImportError: #gcloud
  from dataset import Dataset

class Utils():
  def __init__(self,
        data_path=None):

    if data_path is not None:
      self.ds = Dataset(data_path)
    self.vgg_means = [103.939, 116.779, 123.68] # BGR

  def next_batch_train(self, n_batch=None, width=224, height=224, model='vgg'):
    x_batch = []

    x_train, batch_end = self.ds.get_train_batch(n_batch)
    for x in x_train:
      x_img = self.get_img(x, width=width, height=height, model=model)
      x_batch.append(x_img)

    x_batch = np.array(x_batch)

    return x_batch.astype(np.float32), batch_end

  def next_batch_val(self, n_batch=None, width=224, height=224, model='vgg'):
    x_batch = []

    x_val = self.ds.get_val_batch(n_batch)
    for x in x_val:
      x_img = self.get_img(x, width=width, height=height, model=model)
      x_batch.append(x_img)

    x_batch = np.array(x_batch)

    return x_batch.astype(np.float32)

  def get_img(self, img_name, width=224, height=224, model='vgg'):
    img = cv2.imread(img_name)

    if width != -1 and height != -1:
      img = self.resize_img(img, width, height)

    img = self.normalize_img(img, model)

    return img

  def resize_img(self, img, width, height):
    img = cv2.resize(img, (width, height))

    return img

  def resize_with_ratio(self, width=None, height=None, size=None):
    if (size is not None or size != -1) and height > width:
      new_height = size
      new_width = int((width * new_height)
                      / height)
    elif (size is not None or size != -1):
      new_width = size
      new_height = int((height * new_width)
                       / width)
    elif height is not None and width is None:
      new_height = height
      new_width = int((width * new_height)
                      / height)
    elif height is None and width is not None:
      new_width = width
      new_height = int((height * new_width)
                       / width)
    elif size is not None:
      new_height = height
      new_width = width
    else:
      new_height = None
      new_width = None

    return new_height, new_width

  def normalize_img(self, img, model='vgg'):
    img = img.astype(np.float32)

    if model == 'vgg':
      img = img - self.vgg_means
    elif model == 'transform_net':
      img = img / 127.5 - 1

    return img

  def denormalize_img(self, img, model='vgg'):
    if model == 'vgg':
      img = img + self.vgg_means
    elif model == 'transform_net':
      img = (img + 1) * 127.5

    img[img > 255.] = 255.
    img[img < 0.] = 0.
    img = img.astype(np.uint8)

    return img

  def show_img(self, img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def save_img(self, img, name_path):
    cv2.imwrite(name_path, img)

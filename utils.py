import cv2
import numpy as np

class Utils():
  def __init__(self):
    self.vgg_means = [103.939, 116.779, 123.68] # BGR

  def get_img(self, img_name, model='vgg'):
    img = cv2.imread(img_name)
    img = self.normalize_img(img, model)

    return img

  def normalize_img(self, img, model='vgg'):
    img = img.astype(np.float32)

    if model == 'vgg':
      img = img - self.vgg_means

    return img

  def denormalize_img(self, img, model='vgg'):
    if model == 'vgg':
      img = img + self.vgg_means

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

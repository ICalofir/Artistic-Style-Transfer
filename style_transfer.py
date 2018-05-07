import cv2
import tensorflow as tf
import numpy as np

from conv_nets.vgg19 import VGG19

class StyleTransfer():
  def __init__(self,
      img_height=224,
      img_width=224,
      img_channels=3):

    self.model = VGG19()
    self.img_height = img_height
    self.img_width = img_width
    self.img_channels = img_channels

  def build(self):
    tf.reset_default_graph()

    self.x = tf.placeholder(tf.float32, [None, self.img_height,
                self.img_width, self.img_channels])

    self.out, self.layers = self.model.run(self.x)

  def train(self):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      img = cv2.imread('lena.jpg')
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = np.reshape(img, (1, 224, 224, 3))

      img_r, layer = sess.run([self.out, self.layers], feed_dict={self.x: img})
      # print(img_r)
      print(layer['conv1_1'])

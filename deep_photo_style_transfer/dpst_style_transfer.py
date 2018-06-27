import cv2
import numpy as np
import scipy.io as sio
import tensorflow as tf
from io import BytesIO
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from utils import Utils
try:
  from conv_nets.vgg19 import VGG19
except ImportError: #gcloud
  from vgg19 import VGG19

class StyleTransfer():
  def __init__(self,
      model_name='vgg19',
      tensorflow_model_path='pretrained_models/vgg19/model/tensorflow/conv_wb.pkl',
      content_img_height=224,
      content_img_width=224,
      content_img_channels=3,
      style_img_height=224,
      style_img_width=224,
      style_img_channels=3,
      noise_img_height=224,
      noise_img_width=224,
      noise_img_channels=3,
      content_layers=['relu4_2'],
      style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
      style_layers_w=[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
      alfa=100.0,
      beta=1.0,
      gamma=0.03,
      learning_rate=2,
      num_iters=2000,
      output_img_init='random'):
    self.vgg_means = [103.939, 116.779, 123.68] # BGR
    self.model_name = model_name
    self.tensorflow_model_path = tensorflow_model_path

    self.content_img_height = content_img_height
    self.content_img_width = content_img_width
    self.content_img_channels = content_img_channels
    self.style_img_height = style_img_height
    self.style_img_width = style_img_width
    self.style_img_channels = style_img_channels
    self.noise_img_height = noise_img_height
    self.noise_img_width = noise_img_width
    self.noise_img_channels = noise_img_channels

    self.content_layers = content_layers
    self.style_layers = style_layers
    self.style_layers_w = style_layers_w
    self.alfa = alfa
    self.beta = beta
    self.gamma = gamma

    self.learning_rate = learning_rate
    self.num_iters = num_iters

    self.output_img_init = output_img_init

    # self.mask_color = {'color': [B, G, R]}
    self.mask_color = {'white': [205, 205, 205],
                       'black': [52, 52, 52],
                       'blue': [205, 52, 52],
                       'green': [52, 205, 52],
                       'red': [52, 52, 205]}
    self.mask_channels = len(self.mask_color)

  def _get_mask_img(self, mask_img_path,
                    mask_img_height,
                    mask_img_width,
                    mask_img_channels):
    ut = Utils()
    mask_img = ut.get_img(mask_img_path,
                          width=mask_img_width,
                          height=mask_img_height,
                          model=None)
    input_mask = None

    for k, v in self.mask_color.items():
      curr_mask = np.zeros((mask_img.shape[0], mask_img.shape[1], 1), dtype=np.uint8)

      for i in range(mask_img.shape[0]):
        for j in range((mask_img.shape[1])):
          if k == 'white' \
                  and mask_img[i][j][0] > v[0] \
                  and mask_img[i][j][1] > v[1] \
                  and mask_img[i][j][2] > v[2]:
            curr_mask[i][j][0] = 1
          elif k == 'black' \
                    and mask_img[i][j][0] < v[0] \
                    and mask_img[i][j][1] < v[1] \
                    and mask_img[i][j][2] < v[2]:
            curr_mask[i][j][0] = 1
          elif k == 'blue' \
                    and mask_img[i][j][0] > v[0] \
                    and mask_img[i][j][1] < v[1] \
                    and mask_img[i][j][2] < v[2]:
            curr_mask[i][j][0] = 1
          elif k == 'green' \
                    and mask_img[i][j][0] < v[0] \
                    and mask_img[i][j][1] > v[1] \
                    and mask_img[i][j][2] < v[2]:
            curr_mask[i][j][0] = 1
          elif k == 'red' \
                    and mask_img[i][j][0] < v[0] \
                    and mask_img[i][j][1] < v[1] \
                    and mask_img[i][j][2] > v[2]:
            curr_mask[i][j][0] = 1

      if input_mask is None:
        input_mask = curr_mask
      else:
        input_mask = np.concatenate((input_mask, curr_mask), axis=2)

    return input_mask

  def _get_content_loss(self, content_layer, noise_layer):
    content_loss = tf.constant(0.0)
    content_loss = content_loss \
                   + tf.reduce_sum(tf.square(content_layer \
                                             - noise_layer))
    content_loss = tf.scalar_mul(1.0 / (2.0
                                 * tf.cast(tf.shape(content_layer)[1],
                                           tf.float32)
                                 * tf.cast(tf.shape(content_layer)[2],
                                           tf.float32)
                                 * tf.cast(tf.shape(content_layer)[3],
                                           tf.float32)),
                                 content_loss)

    return content_loss

  def _get_style_loss(self, style_layer, noise_layer,
                            mask_style_img, mask_content_img):
    style_loss = tf.constant(0.0)

    sz = tf.constant([style_layer.get_shape().as_list()[1],
                      style_layer.get_shape().as_list()[2]])
    style_img = tf.image.resize_images(mask_style_img, sz)
    sz = tf.constant([noise_layer.get_shape().as_list()[1],
                      noise_layer.get_shape().as_list()[2]])
    content_img = tf.image.resize_images(mask_content_img, sz)

    for c in range(self.mask_channels):
      mask_matrix_style_img = tf.multiply(style_layer,
                                          tf.reshape(style_img[:, :, :, c],
                                                     [tf.shape(style_img)[0],
                                                      tf.shape(style_img)[1],
                                                      tf.shape(style_img)[2],
                                                      1]))
      mask_matrix_noise_img = tf.multiply(noise_layer,
                                          tf.reshape(content_img[:, :, :, c],
                                                     [tf.shape(content_img)[0],
                                                      tf.shape(content_img)[1],
                                                      tf.shape(content_img)[2],
                                                      1]))

      channels_matrix_style_img = tf.reshape(mask_matrix_style_img,
                                             [-1, tf.shape(mask_matrix_style_img)[3]])
      channels_matrix_noise_img = tf.reshape(mask_matrix_noise_img,
                                             [-1, tf.shape(mask_matrix_noise_img)[3]])

      gram_matrix_style = tf.matmul(tf.transpose(channels_matrix_style_img),
                                    channels_matrix_style_img)
      gram_matrix_noise = tf.matmul(tf.transpose(channels_matrix_noise_img),
                                    channels_matrix_noise_img)

      El = tf.reduce_sum(tf.square(gram_matrix_style - gram_matrix_noise))
      El = tf.scalar_mul(1.0 / (4.0
                         * tf.square(tf.cast(tf.shape(style_layer)[1],
                                             tf.float32))
                         * tf.square(tf.cast(tf.shape(style_layer)[2],
                                             tf.float32))
                         * tf.square(tf.cast(tf.shape(style_layer)[3],
                                             tf.float32))),
                         El)
      style_loss = style_loss + El

    return style_loss

  def _get_total_variation_loss(self, noise_img):
    tv_loss = tf.reduce_sum(tf.image.total_variation(noise_img))

    return tv_loss

  def build(self):
    tf.reset_default_graph()

    if self.model_name == 'vgg19':
      self.model = VGG19(tensorflow_model_path=self.tensorflow_model_path)
    self.content_img = tf.placeholder(tf.float32, [None, self.content_img_height,
                self.content_img_width, self.content_img_channels])
    self.style_img = tf.placeholder(tf.float32, [None, self.style_img_height,
                self.style_img_width, self.style_img_channels])

    self.mask_content_img = tf.placeholder(tf.float32, [None, self.content_img_height,
                self.content_img_width, self.mask_channels])
    self.mask_style_img = tf.placeholder(tf.float32, [None, self.style_img_height,
                self.style_img_width, self.mask_channels])

    self.noise_img_init = tf.placeholder(tf.float32, [1, self.noise_img_height,
                self.noise_img_width, self.noise_img_channels])

    if self.output_img_init == 'content':
      self.noise_img = tf.get_variable(name='output_image',
                                       initializer=self.noise_img_init)
    elif self.output_img_init == 'random':
      # xavier init
      self.noise_img = tf.get_variable(name='output_image',
                                       shape=[1,
                                              self.noise_img_height,
                                              self.noise_img_width,
                                              self.noise_img_channels],
                                       initializer=None)

    self.total_variation_loss = self._get_total_variation_loss(self.noise_img)

    self.content_loss = tf.constant(0.0)
    for content_layer_name in self.content_layers:
      content_layer = self.model.run(self.content_img, content_layer_name)
      noise_layer = self.model.run(self.noise_img, content_layer_name)
      self.content_loss = self.content_loss \
                          + self._get_content_loss(content_layer, noise_layer)

    self.style_loss = tf.constant(0.0)
    for i, style_layer_name in enumerate(self.style_layers):
      style_layer = self.model.run(self.style_img, style_layer_name)
      noise_layer = self.model.run(self.noise_img, style_layer_name)
      self.style_loss = self.style_loss \
                          + tf.scalar_mul(self.style_layers_w[i],
                                self._get_style_loss(style_layer, noise_layer,
                                                     self.mask_style_img,
                                                     self.mask_content_img))

    self.noise_img_normalized = (self.noise_img + self.vgg_means) / 255.0

    self.total_loss = self.alfa * self.content_loss \
                      + self.beta * self.style_loss \
                      + self.gamma * self.total_variation_loss

    var_list = tf.trainable_variables()
    self.var_list = [var for var in var_list if 'output_image' in var.name]

    self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        name='adam_optimizer').minimize(self.total_loss, var_list=self.var_list)

    tf.summary.scalar('content_loss', self.content_loss)
    tf.summary.scalar('style_loss', self.style_loss)
    tf.summary.scalar('tv_loss', self.total_variation_loss)
    tf.summary.scalar('total_loss', self.total_loss)

    tf.summary.histogram("noise_img", self.noise_img)

    self.decoded_img = tf.placeholder(tf.uint8,
                                      [self.noise_img_height,
                                       self.noise_img_width,
                                       self.noise_img_channels])
    self.name_file = tf.placeholder(tf.string)

    self.encoded_img = tf.image.encode_png(self.decoded_img)
    self.fwrite = tf.write_file(self.name_file, self.encoded_img)

    self.file_bytes = tf.read_file(self.name_file)

  def train(self,
            content_img_path='images/content/d_content6_resized.png',
            style_img_path='images/content/d_content6_resized.png',
            noise_img_path='images/content/d_content6_resized.png',
            mask_content_img_path='images/mask/mask_d_content6_resized.png',
            mask_style_img_path='images/mask/mask_d_style6_resized.png',
            output_img_path='results/dpst',
            tensorboard_path='tensorboard/tensorboard_dpst',
            show_img=None):
    summ = tf.summary.merge_all()
    with tf.Session() as sess:
      ut = Utils()
      noise_img_bytes = sess.run(self.file_bytes,
          feed_dict={self.name_file: noise_img_path})
      noise_img_np = np.fromstring(noise_img_bytes, np.uint8)
      noise_img = np.reshape(ut.add_noise(ut.get_img(noise_img_np,
                                                     width=self.noise_img_width,
                                                     height=self.noise_img_height)),
                             (1,
                             self.noise_img_height,
                             self.noise_img_width,
                             self.noise_img_channels))
      sess.run(tf.global_variables_initializer(), feed_dict={self.noise_img_init: noise_img})

      writer = tf.summary.FileWriter(tensorboard_path)
      writer.add_graph(sess.graph)

      img_bytes = sess.run(self.file_bytes,
          feed_dict={self.name_file: content_img_path})
      img_np = np.fromstring(img_bytes, np.uint8)
      content_img = np.reshape(ut.get_img(img_np,
                                          width=self.content_img_width,
                                          height=self.content_img_height),
                                          (1,
                                           self.content_img_height,
                                           self.content_img_width,
                                           self.content_img_channels))
      img_bytes = sess.run(self.file_bytes,
          feed_dict={self.name_file: style_img_path})
      img_np = np.fromstring(img_bytes, np.uint8)
      style_img = np.reshape(ut.get_img(img_np,
                                        width=self.style_img_width,
                                        height=self.style_img_height),
                                        (1,
                                         self.style_img_height,
                                         self.style_img_width,
                                         self.style_img_channels))

      img_bytes = sess.run(self.file_bytes,
          feed_dict={self.name_file: mask_content_img_path})
      img_np = np.fromstring(img_bytes, np.uint8)
      mask_content_img = np.reshape(self._get_mask_img(img_np,
                                                       self.content_img_height,
                                                       self.content_img_width,
                                                       self.content_img_channels),
                                    (1,
                                     self.content_img_height,
                                     self.content_img_width,
                                     self.mask_channels))
      img_bytes = sess.run(self.file_bytes,
          feed_dict={self.name_file: mask_style_img_path})
      img_np = np.fromstring(img_bytes, np.uint8)
      mask_style_img = np.reshape(self._get_mask_img(img_np,
                                                     self.style_img_height,
                                                     self.style_img_width,
                                                     self.style_img_channels),
                                  (1,
                                   self.style_img_height,
                                   self.style_img_width,
                                   self.mask_channels))
      print('Done loading mask images.')

      for i in range(self.num_iters):
        _, content_loss, style_loss, tv_loss, out_loss, out_img =  sess.run(
              [self.optim,
               self.content_loss,
               self.style_loss,
               self.total_variation_loss,
               self.total_loss,
               self.noise_img],
              feed_dict={self.content_img: content_img,
                         self.style_img: style_img,
                         self.mask_content_img: mask_content_img,
                         self.mask_style_img: mask_style_img})

        print('it: ', i)
        print('Content loss: ', content_loss)
        print('Style loss: ', style_loss)
        print('Total variation loss: ', tv_loss)
        print('Total loss: ', out_loss)

        if i % 50 == 0:
          decoded_img = ut.denormalize_img(out_img[0])
          decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
          sess.run(self.fwrite,
                   feed_dict={self.decoded_img: decoded_img,
                              self.name_file: output_img_path + '/img' + str(i) + '.png'})

          if show_img:
            plt.axis("off")
            plt.imshow(decoded_img)
            plt.show()

          s = sess.run(summ,
                       feed_dict={self.content_img: content_img,
                                  self.style_img: style_img,
                                  self.mask_content_img: mask_content_img,
                                  self.mask_style_img: mask_style_img})
          writer.add_summary(s, i)

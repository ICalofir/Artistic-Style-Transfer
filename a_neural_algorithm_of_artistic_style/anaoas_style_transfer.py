import cv2
import numpy as np
import tensorflow as tf

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
      alfa=1.0,
      beta=100.0,
      gamma=0.001,
      learning_rate=2,
      num_iters=2000,
      output_img_init='random'):
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

  def _get_style_loss(self, style_layer, noise_layer):
    style_loss = tf.constant(0.0)

    channels_matrix_style_img = tf.reshape(style_layer,
                                           [-1, tf.shape(style_layer)[3]])
    channels_matrix_noise_img = tf.reshape(noise_layer,
                                           [-1, tf.shape(noise_layer)[3]])

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
                                self._get_style_loss(style_layer, noise_layer))

    if self.gamma == 0.0:
      self.total_loss = self.alfa * self.content_loss \
                        + self.beta * self.style_loss
    else:
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
            content_img_path='images/content/content1.jpg',
            style_img_path='images/style/style1.jpg',
            noise_img_path='images/content/content1.jpg',
            output_img_path='results/anaoas',
            tensorboard_path='tensorboard/tensorboard_anaoas'):
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

      content_img_bytes = sess.run(self.file_bytes,
          feed_dict={self.name_file: content_img_path})
      content_img_np = np.fromstring(content_img_bytes, np.uint8)
      content_img = np.reshape(ut.get_img(content_img_np,
                                          width=self.content_img_width,
                                          height=self.content_img_height),
                                          (1,
                                           self.content_img_height,
                                           self.content_img_width,
                                           self.content_img_channels))
      style_img_bytes = sess.run(self.file_bytes,
          feed_dict={self.name_file: style_img_path})
      style_img_np = np.fromstring(style_img_bytes, np.uint8)
      style_img = np.reshape(ut.get_img(style_img_np,
                                        width=self.style_img_width,
                                        height=self.style_img_height),
                                        (1,
                                         self.style_img_height,
                                         self.style_img_width,
                                         self.style_img_channels))

      for i in range(self.num_iters):
        _, content_loss, style_loss, tv_loss, out_loss, out_img =  sess.run(
              [self.optim, self.content_loss, self.style_loss, self.total_variation_loss,
               self.total_loss, self.noise_img],
              feed_dict={self.content_img: content_img,
                         self.style_img: style_img})

        print('it: ', i)
        print('Content loss: ', content_loss)
        print('Style loss: ', style_loss)
        print('Total variation loss: ', tv_loss)
        print('Total loss: ', out_loss)

        if i % 100 == 0:
          decoded_img = ut.denormalize_img(out_img[0])
          decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
          sess.run(self.fwrite,
                   feed_dict={self.decoded_img: decoded_img,
                              self.name_file: output_img_path + '/img' + str(i) + '.png'})

          s = sess.run(summ,
                       feed_dict={self.content_img: content_img,
                                  self.style_img: style_img})
          writer.add_summary(s, i)

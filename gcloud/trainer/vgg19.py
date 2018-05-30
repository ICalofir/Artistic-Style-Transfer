import cv2
import pickle
import numpy as np
import tensorflow as tf

class VGG19():
  def __init__(self,
      tensorflow_model_path=
        'pretrained_models/vgg19/model/tensorflow/conv_wb.pkl'):

      self.tensorflow_model_path = tensorflow_model_path
      # with open(self.tensorflow_model_path, 'rb') as f:
        # self.tensorflow_model = pickle.load(f)
      s = tf.InteractiveSession()
      fread = tf.read_file(self.tensorflow_model_path)
      self.tensorflow_model = pickle.loads(fread.eval())
      s.close()

  def run(self, img, layer_name, name='vgg19'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      # conv1_1
      with tf.variable_scope('conv1_1'):
        net = tf.contrib.layers.conv2d(img, 64, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv1_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv1_1']['biases']),
                  activation_fn=None)
        if layer_name == 'conv1_1':
          return net
        net = tf.nn.relu(net)

      # conv1_2
      with tf.variable_scope('conv1_2'):
        net = tf.contrib.layers.conv2d(net, 64, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv1_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv1_2']['biases']),
                  activation_fn=None)
        if layer_name == 'conv1_2':
          return net
        net = tf.nn.relu(net)

      # maxpool
      with tf.variable_scope('pool1'):
        net = tf.contrib.layers.avg_pool2d(net, 2)

      # conv2_1
      with tf.variable_scope('conv2_1'):
        net = tf.contrib.layers.conv2d(net, 128, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv2_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv2_1']['biases']),
                  activation_fn=None)
        if layer_name == 'conv2_1':
          return net
        net = tf.nn.relu(net)

      # conv2_2
      with tf.variable_scope('conv2_2'):
        net = tf.contrib.layers.conv2d(net, 128, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv2_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv2_2']['biases']),
                  activation_fn=None)
        if layer_name == 'conv2_2':
          return net
        net = tf.nn.relu(net)

      # maxpool
      with tf.variable_scope('pool2'):
        net = tf.contrib.layers.avg_pool2d(net, 2)

      # conv3_1
      with tf.variable_scope('conv3_1'):
        net = tf.contrib.layers.conv2d(net, 256, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_1']['biases']),
                  activation_fn=None)
        if layer_name == 'conv3_1':
          return net
        net = tf.nn.relu(net)

      # conv3_2
      with tf.variable_scope('conv3_2'):
        net = tf.contrib.layers.conv2d(net, 256, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_2']['biases']),
                  activation_fn=None)
        if layer_name == 'conv3_2':
          return net
        net = tf.nn.relu(net)

      # conv3_3
      with tf.variable_scope('conv3_3'):
        net = tf.contrib.layers.conv2d(net, 256, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_3']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_3']['biases']),
                  activation_fn=None)
        if layer_name == 'conv3_3':
          return net
        net = tf.nn.relu(net)

      # conv3_4
      with tf.variable_scope('conv3_4'):
        net = tf.contrib.layers.conv2d(net, 256, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_4']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_4']['biases']),
                  activation_fn=None)
        if layer_name == 'conv3_4':
          return net
        net = tf.nn.relu(net)

      # maxpool
      with tf.variable_scope('pool3'):
        net = tf.contrib.layers.avg_pool2d(net, 2)

      # conv4_1
      with tf.variable_scope('conv4_1'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_1']['biases']),
                  activation_fn=None)
        if layer_name == 'conv4_1':
          return net
        net = tf.nn.relu(net)

      # conv4_2
      with tf.variable_scope('conv4_2'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_2']['biases']),
                  activation_fn=None)
        if layer_name == 'conv4_2':
          return net
        net = tf.nn.relu(net)

      # conv4_3
      with tf.variable_scope('conv4_3'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_3']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_3']['biases']),
                  activation_fn=None)
        if layer_name == 'conv4_3':
          return net
        net = tf.nn.relu(net)

      # conv4_4
      with tf.variable_scope('conv4_4'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_4']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_4']['biases']),
                  activation_fn=None)
        if layer_name == 'conv4_4':
          return net
        net = tf.nn.relu(net)

      # maxpool
      with tf.variable_scope('pool4'):
        net = tf.contrib.layers.avg_pool2d(net, 2)

      # conv5_1
      with tf.variable_scope('conv5_1'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_1']['biases']),
                  activation_fn=None)
        if layer_name == 'conv5_1':
          return net
        net = tf.nn.relu(net)

      # conv5_2
      with tf.variable_scope('conv5_2'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_2']['biases']),
                  activation_fn=None)
        if layer_name == 'conv5_2':
          return net
        net = tf.nn.relu(net)

      # conv5_3
      with tf.variable_scope('conv5_3'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_3']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_3']['biases']),
                  activation_fn=None)
        if layer_name == 'conv5_3':
          return net
        net = tf.nn.relu(net)

      # conv5_4
      with tf.variable_scope('conv5_4'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_4']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_4']['biases']),
                  activation_fn=None)
        if layer_name == 'conv5_4':
          return net
        net = tf.nn.relu(net)

      # maxpool
      with tf.variable_scope('pool5'):
        net = tf.contrib.layers.avg_pool2d(net, 2)

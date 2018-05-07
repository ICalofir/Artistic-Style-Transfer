import cv2
import pickle
import tensorflow as tf
import numpy as np

class VGG19():
  def __init__(self,
      tensorflow_model_path=
        'pretrained_models/vgg19/model/tensorflow/conv_wb.pkl'):

      self.tensorflow_model_path = tensorflow_model_path
      with open(self.tensorflow_model_path, 'rb') as f:
        self.tensorflow_model = pickle.load(f)

  def run(self, img, name='vgg19'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      layers = {}

      # conv1_1
      with tf.variable_scope('conv1_1'):
        net = tf.contrib.layers.conv2d(img, 64, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv1_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv1_1']['biases']))
        layers['conv1_1'] = net

      # conv1_2
      with tf.variable_scope('conv1_2'):
        net = tf.contrib.layers.conv2d(net, 64, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv1_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv1_2']['biases']))
        layers['conv1_2'] = net

      # maxpool
      with tf.variable_scope('pool1'):
        net = tf.contrib.layers.max_pool2d(net, 2)
        layers['pool1'] = net

      # conv2_1
      with tf.variable_scope('conv2_1'):
        net = tf.contrib.layers.conv2d(net, 128, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv2_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv2_1']['biases']))
        layers['conv2_1'] = net

      # conv2_2
      with tf.variable_scope('conv2_2'):
        net = tf.contrib.layers.conv2d(net, 128, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv2_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv2_2']['biases']))
        layers['conv2_2'] = net

      # maxpool
      with tf.variable_scope('pool2'):
        net = tf.contrib.layers.max_pool2d(net, 2)
        layers['pool2'] = net

      # conv3_1
      with tf.variable_scope('conv3_1'):
        net = tf.contrib.layers.conv2d(net, 256, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_1']['biases']))
        layers['conv3_1'] = net

      # conv3_2
      with tf.variable_scope('conv3_2'):
        net = tf.contrib.layers.conv2d(net, 256, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_2']['biases']))
        layers['conv3_2'] = net

      # conv3_3
      with tf.variable_scope('conv3_3'):
        net = tf.contrib.layers.conv2d(net, 256, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_3']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_3']['biases']))
        layers['conv3_3'] = net

      # conv3_4
      with tf.variable_scope('conv3_4'):
        net = tf.contrib.layers.conv2d(net, 256, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_4']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv3_4']['biases']))
        layers['conv3_4'] = net

      # maxpool
      with tf.variable_scope('pool3'):
        net = tf.contrib.layers.max_pool2d(net, 2)
        layers['pool3'] = net

      # conv4_1
      with tf.variable_scope('conv4_1'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_1']['biases']))
        layers['conv4_1'] = net

      # conv4_2
      with tf.variable_scope('conv4_2'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_2']['biases']))
        layers['conv4_2'] = net

      # conv4_3
      with tf.variable_scope('conv4_3'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_3']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_3']['biases']))
        layers['conv4_1'] = net

      # conv4_4
      with tf.variable_scope('conv4_4'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_4']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv4_4']['biases']))
        layers['conv4_1'] = net

      # maxpool
      with tf.variable_scope('pool4'):
        net = tf.contrib.layers.max_pool2d(net, 2)
        layers['pool4'] = net

      # conv5_1
      with tf.variable_scope('conv5_1'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_1']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_1']['biases']))
        layers['conv5_1'] = net

      # conv5_2
      with tf.variable_scope('conv5_2'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_2']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_2']['biases']))
        layers['conv5_2'] = net

      # conv5_3
      with tf.variable_scope('conv5_3'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_3']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_3']['biases']))
        layers['conv5_3'] = net

      # conv5_4
      with tf.variable_scope('conv5_4'):
        net = tf.contrib.layers.conv2d(net, 512, 3,
                  weights_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_4']['weights']),
                  biases_initializer=tf.constant_initializer(
                      self.tensorflow_model['conv5_4']['biases']))
        layers['conv5_4'] = net

      # maxpool
      with tf.variable_scope('pool5'):
        net = tf.contrib.layers.max_pool2d(net, 2)
        layers['pool5'] = net

    return net, layers

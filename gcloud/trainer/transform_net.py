import tensorflow as tf

class TransformNet():
  def __init__(self):
    pass

  def run(self, img, name='transform_net'):
    with tf.variable_scope(name):

      # conv1
      with tf.variable_scope('conv1'):
        net = tf.contrib.layers.conv2d(img, 32, 9, activation_fn=None)
        net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)

        tf.summary.histogram("activation", net)

      # conv2
      with tf.variable_scope('conv2'):
        net = tf.contrib.layers.conv2d(net, 64, 3, stride=2, activation_fn=None)
        net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)

        tf.summary.histogram("activation", net)

      # conv3
      with tf.variable_scope('conv3'):
        net = tf.contrib.layers.conv2d(net, 128, 3, stride=2, activation_fn=None)
        net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)

        tf.summary.histogram("activation", net)

      # res_block1
      with tf.variable_scope('res_block1'):
        res_net = tf.contrib.layers.conv2d(net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        res_net = tf.nn.relu(res_net)
        res_net = tf.contrib.layers.conv2d(res_net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        net = net + res_net

        tf.summary.histogram("activation", net)

      # res_block2
      with tf.variable_scope('res_block2'):
        res_net = tf.contrib.layers.conv2d(net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        res_net = tf.nn.relu(res_net)
        res_net = tf.contrib.layers.conv2d(res_net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        net = net + res_net

        tf.summary.histogram("activation", net)

      # res_block3
      with tf.variable_scope('res_block3'):
        res_net = tf.contrib.layers.conv2d(net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        res_net = tf.nn.relu(res_net)
        res_net = tf.contrib.layers.conv2d(res_net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        net = net + res_net

        tf.summary.histogram("activation", net)

      # res_block4
      with tf.variable_scope('res_block4'):
        res_net = tf.contrib.layers.conv2d(net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        res_net = tf.nn.relu(res_net)
        res_net = tf.contrib.layers.conv2d(res_net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        net = net + res_net

        tf.summary.histogram("activation", net)

      # res_block5
      with tf.variable_scope('res_block5'):
        res_net = tf.contrib.layers.conv2d(net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        res_net = tf.nn.relu(res_net)
        res_net = tf.contrib.layers.conv2d(res_net, 128, 3, activation_fn=None)
        res_net = tf.contrib.layers.batch_norm(net)
        net = net + res_net

        tf.summary.histogram("activation", net)

      # conv_transpose1
      with tf.variable_scope('conv_transpose1'):
        net = tf.contrib.layers.conv2d_transpose(net, 64, 3, stride=2, activation_fn=None)
        net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)

        tf.summary.histogram("activation", net)

      # conv_transpose2
      with tf.variable_scope('conv_transpose2'):
        net = tf.contrib.layers.conv2d_transpose(net, 32, 3, stride=2, activation_fn=None)
        net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)

        tf.summary.histogram("activation", net)

      # conv_transpose3
      with tf.variable_scope('conv_transpose3'):
        net = tf.contrib.layers.conv2d_transpose(net, 3, 9, activation_fn=tf.nn.tanh)

        tf.summary.histogram("activation", net)

    return net

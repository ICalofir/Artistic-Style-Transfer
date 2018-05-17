import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
import tensorflow as tf

x = tf.sparse_placeholder(tf.float32)
z = tf.constant([1., 1., 1., 1., 1., 1.], shape=[6, 1], dtype=tf.float32)
w = tf.constant([2., 2., 2., 2., 2., 2.], shape=[1, 6], dtype=tf.float32)
y = tf.sparse_tensor_dense_matmul(x, z)
y = tf.matmul(w, y)
y = tf.reduce_sum(y)

with tf.Session() as sess:
  mat = sio.loadmat('test.mat')
  print(mat)

  sp = mat['mat'][0][0][0]
  print(sp)

  print(sp[0:2, 2])
  row_ind = sp[:, 0].astype(np.int64)
  col_ind = sp[:, 1].astype(np.int64)
  ind = []
  for i in range(len(row_ind)):
    ind.append([row_ind[i] - 1, col_ind[i] - 1]) # -1 because octave starts from 1
  val = sp[:, 2].astype(np.float32)
  h = 6
  w = 6
  print(row_ind)
  print(col_ind)
  print(ind)
  print(val)
  print(h, w)

  indices = np.array(ind, dtype=np.int64)
  values = np.array(val, dtype=np.float32)
  shape = np.array([h, w], dtype=np.int64)
  print(indices)
  print(values)
  print(shape)

  print(sess.run(y, feed_dict={
    x: tf.SparseTensorValue(indices, values, shape)}))

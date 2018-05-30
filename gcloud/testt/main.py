import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path',
                    help='')

args = parser.parse_args()

path = args.path
# print(path)
# print(type(path))
# img = cv2.imread(path)
# print(img)

path = '../../images/content/content1'
path_p = '../../pretrained_models/vgg19/model/tensorflow/conv_wb.pkl'

import tensorflow as tf
import pickle

# image_decoded = tf.image.decode_jpeg(tf.read_file(path + '.jpg'), channels=3)
# cropped       = tf.image.resize_image_with_crop_or_pad(image_decoded, 200, 200)
# image_decoded = tf.placeholder(tf.uint8, shape=(525, 700, 3))

# enc = tf.image.encode_jpeg(image_decoded)
# fname = tf.constant(path + '22.jpg')
# fwrite = tf.write_file(fname, enc)

fread = tf.read_file(path_p)

s = tf.InteractiveSession()
aa = pickle.loads(fread.eval())
s.close()

with open(path_p, 'rb') as f:
  t = pickle.load(f)

caca = tf.constant(aa['conv1_1']['weights'])
s = tf.InteractiveSession()
fu = caca.eval()
print(fu)
s.close()

sess = tf.Session()
# lala = cv2.imread(path + '.jpg')
# lala = cv2.cvtColor(lala, cv2.COLOR_BGR2RGB)
# print(lala.shape)
# result = sess.run(fwrite, feed_dict={image_decoded: lala})
result = sess.run(fread)

# print(sess.graph)
plm = tf.Session()
# cc = plm.run(aa)
# print(plm.graph)
# print(type(result))
# p = pickle.loads(result)
# print(p)
# print(img.shape)
# print(type(img))
sess.close()

# print(cc)

p = pickle.loads(result)


with open(path_p, 'rb') as f:
  t = pickle.load(f)
  # print(t)

import numpy as np
print(np.sum(fu - t['conv1_1']['weights']))
# print(p.keys())
# print(t.keys())

# print(p['conv1_1']['weights'].shape)
# print(np.sum(p['conv2_1']['weights'] - t['conv2_1']['weights']))

# tf.gfile.MakeDirs('laaa/yuhuuu')

def fc():
  s = tf.InteractiveSession()
  s.close

with tf.Session() as ss:
  print(ss.run(caca))
  print('doone')

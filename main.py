import argparse
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf

from utils import Utils
try:
  import a_neural_algorithm_of_artistic_style.anaoas_style_transfer as anaoas
  import perceptual_losses_for_real_time_style_transfer.plfrtst_style_transfer as plfrtst
  import deep_photo_style_transfer.dpst_style_transfer as dpst
except ImportError: #gcloud
  import anaoas_style_transfer as anaoas
  import plfrtst_style_transfer as plfrtst
  import dpst_style_transfer as dpst

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--method',
                      help='',
                      required=True)
  parser.add_argument('--train',
                      help='if true, start training a neural network',
                      action='store_true')
  parser.add_argument('--model_name',
                      help='')
  parser.add_argument('--tensorflow_model_path',
                      help='')
  parser.add_argument('--data_path',
                      help='')
  parser.add_argument('--content_img_size',
                      help='keep ratio of image and max(widht, height) will be equal to size',
                      type=int)
  parser.add_argument('--content_img_width',
                      help='',
                      type=int)
  parser.add_argument('--content_img_height',
                      help='',
                      type=int)
  parser.add_argument('--content_img_channels',
                      help='',
                      type=int)
  parser.add_argument('--style_img_size',
                      help='keep ratio of image and max(widht, height) will be equal to size',
                      type=int)
  parser.add_argument('--style_img_width',
                      help='',
                      type=int)
  parser.add_argument('--style_img_height',
                      help='',
                      type=int)
  parser.add_argument('--style_img_channels',
                      help='',
                      type=int)
  parser.add_argument('--content_layers',
                      help='',
                      nargs='+')
  parser.add_argument('--style_layers',
                      help='',
                      nargs='+')
  parser.add_argument('--style_layers_w',
                      help='',
                      nargs='+',
                      type=float)
  parser.add_argument('--alfa',
                      help='',
                      type=float)
  parser.add_argument('--beta',
                      help='',
                      type=float)
  parser.add_argument('--gamma',
                      help='',
                      type=float)
  parser.add_argument('--learning_rate',
                      help='',
                      type=float)
  parser.add_argument('--num_iters',
                      help='',
                      type=int)
  parser.add_argument('--batch_size',
                      help='',
                      type=int)
  parser.add_argument('--no_epochs',
                      help='',
                      type=int)
  parser.add_argument('--content_img_path',
                      help='')
  parser.add_argument('--style_img_path',
                      help='')
  parser.add_argument('--mask_content_img_path',
                      help='')
  parser.add_argument('--mask_style_img_path',
                      help='')
  parser.add_argument('--laplacian_matrix_path',
                      help='')
  parser.add_argument('--output_img_path',
                      help='')
  parser.add_argument('--tensorboard_path',
                      help='')
  parser.add_argument('--model_path',
                      help='')

  args = parser.parse_args()

  if args.method == 'anaoas':
    if args.train:
      ut = Utils()
      content_img_path = args.content_img_path or 'images/content/content1.jpg'
      style_img_path = args.style_img_path or 'images/style/style1.jpg'

      s = tf.InteractiveSession()
      content_img_bytes = tf.read_file(content_img_path)
      style_img_bytes = tf.read_file(style_img_path)

      content_img_np = np.fromstring(content_img_bytes.eval(), np.uint8)
      style_img_np = np.fromstring(style_img_bytes.eval(), np.uint8)
      s.close()

      content_img = ut.get_img(content_img_np,
                               width=-1,
                               height=-1)
      style_img = ut.get_img(style_img_np,
                             width=-1,
                             height=-1)
      content_img_height = content_img.shape[0]
      content_img_width = content_img.shape[1]
      style_img_height = style_img.shape[0]
      style_img_width = style_img.shape[1]

      args.content_img_height, args.content_img_width = \
          ut.resize_with_ratio(height=args.content_img_height or content_img_height,
                               width=args.content_img_width or content_img_width,
                               size=args.content_img_size)
      args.style_img_height, args.style_img_width = \
          ut.resize_with_ratio(height=args.style_img_height or style_img_height,
                               width=args.style_img_width or style_img_width,
                               size=args.style_img_size)

      model = anaoas.StyleTransfer(
          model_name=args.model_name or 'vgg19',
          tensorflow_model_path=args.tensorflow_model_path
              or 'pretrained_models/vgg19/model/tensorflow/conv_wb.pkl',
          content_img_height=args.content_img_height or 224,
          content_img_width=args.content_img_width or 224,
          content_img_channels=args.content_img_channels or 3,
          style_img_height=args.style_img_height or 224,
          style_img_width=args.style_img_width or 224,
          style_img_channels=args.style_img_channels or 3,
          noise_img_height=args.content_img_height or 224,
          noise_img_width=args.content_img_width or 224,
          noise_img_channels=args.content_img_channels or 3,
          content_layers=args.content_layers or ['conv4_2'],
          style_layers=args.style_layers
              or ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
          style_layers_w=args.style_layers_w
              or [1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
          alfa=args.alfa or 1,
          beta=args.beta or 1,
          learning_rate=args.learning_rate or 2,
          num_iters=args.num_iters or 1000)

      tensorboard_path = args.tensorboard_path or 'tensorboard/tensorboard_anaoas'
      if tf.gfile.IsDirectory(tensorboard_path):
        tf.gfile.DeleteRecursively(tensorboard_path)
      tf.gfile.MakeDirs(tensorboard_path)

      output_img_path = args.output_img_path or 'results/anaoas'
      if tf.gfile.IsDirectory(output_img_path):
        tf.gfile.DeleteRecursively(output_img_path)
      tf.gfile.MakeDirs(output_img_path)

      model.build()
      model.train(
          content_img_path=content_img_path,
          style_img_path=style_img_path,
          output_img_path=output_img_path,
          tensorboard_path=tensorboard_path)
    else:
      print('Nothing to be done!')
  elif args.method == 'plfrtst':
    pass
  elif args.method == 'dpst':
    if args.train:
      ut = Utils()
      content_img_path = args.content_img_path or 'images/content/d_content1.png'
      style_img_path = args.style_img_path or 'images/style/d_style1.png'

      s = tf.InteractiveSession()
      content_img_bytes = tf.read_file(content_img_path)
      style_img_bytes = tf.read_file(style_img_path)

      content_img_np = np.fromstring(content_img_bytes.eval(), np.uint8)
      style_img_np = np.fromstring(style_img_bytes.eval(), np.uint8)
      s.close()

      content_img = ut.get_img(content_img_np,
                               width=-1,
                               height=-1)
      style_img = ut.get_img(style_img_np,
                             width=-1,
                             height=-1)
      content_img_height = content_img.shape[0]
      content_img_width = content_img.shape[1]
      style_img_height = style_img.shape[0]
      style_img_width = style_img.shape[1]

      args.content_img_height, args.content_img_width = \
          ut.resize_with_ratio(height=args.content_img_height or content_img_height,
                               width=args.content_img_width or content_img_width,
                               size=args.content_img_size)
      args.style_img_height, args.style_img_width = \
          ut.resize_with_ratio(height=args.style_img_height or style_img_height,
                               width=args.style_img_width or style_img_width,
                               size=args.style_img_size)

      model = dpst.StyleTransfer(
          model_name=args.model_name or 'vgg19',
          tensorflow_model_path=args.tensorflow_model_path
              or 'pretrained_models/vgg19/model/tensorflow/conv_wb.pkl',
          content_img_height=args.content_img_height or 224,
          content_img_width=args.content_img_width or 224,
          content_img_channels=args.content_img_channels or 3,
          style_img_height=args.style_img_height or 224,
          style_img_width=args.style_img_width or 224,
          style_img_channels=args.style_img_channels or 3,
          noise_img_height=args.content_img_height or 224,
          noise_img_width=args.content_img_width or 224,
          noise_img_channels=args.content_img_channels or 3,
          content_layers=args.content_layers or ['conv4_2'],
          style_layers=args.style_layers
              or ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
          style_layers_w=args.style_layers_w
              or [1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
          alfa=args.alfa or 1,
          beta=args.beta or 1,
          gamma=args.gamma or 1,
          learning_rate=args.learning_rate or 2,
          num_iters=args.num_iters or 1000)

      tensorboard_path = args.tensorboard_path or 'tensorboard/tensorboard_dpst'
      if tf.gfile.IsDirectory(tensorboard_path):
        tf.gfile.DeleteRecursively(tensorboard_path)
      tf.gfile.MakeDirs(tensorboard_path)

      output_img_path = args.output_img_path or 'results/dpst'
      if tf.gfile.IsDirectory(output_img_path):
        tf.gfile.DeleteRecursively(output_img_path)
      tf.gfile.MakeDirs(output_img_path)

      model.build()
      model.train(
          content_img_path=content_img_path,
          style_img_path=style_img_path,
          mask_content_img_path=args.mask_content_img_path \
                                  or 'images/mask/mask_d_content1.png',
          mask_style_img_path=args.mask_style_img_path \
                                  or 'images/mask/mask_d_style1.png',
          laplacian_matrix_path=args.laplacian_matrix_path \
                                  or 'images/laplacian/d_laplacian1.mat',
          output_img_path=output_img_path,
          tensorboard_path=tensorboard_path)
  else:
    print('Nothing to be done!')

import argparse
import cv2
import os
import shutil

import a_neural_algorithm_of_artistic_style.anaoas_style_transfer as anaoas
import perceptual_losses_for_real_time_style_transfer.plfrtst_style_transfer as plfrtst
import deep_photo_style_transfer.dpst_style_transfer as dpst

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--method',
                      help='',
                      required=True)
  parser.add_argument('--train',
                      help='if true, start training a neural network',
                      action='store_true')
  parser.add_argument('--grid_search',
                      help='if true, check for best hyperparameters',
                      action='store_true')
  parser.add_argument('--model_name',
                      help='')
  parser.add_argument('--tensorflow_model_path',
                      help='')
  parser.add_argument('--data_path',
                      help='')
  parser.add_argument('--content_img_width',
                      help='',
                      type=int)
  parser.add_argument('--content_img_height',
                      help='',
                      type=int)
  parser.add_argument('--content_img_channels',
                      help='',
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
          num_iters=args.num_iters or 2000)

      tensorboard_path = args.tensorboard_path or 'tensorboard/tensorboard_anaoas'
      if os.path.exists(tensorboard_path):
        shutil.rmtree(tensorboard_path)
      os.makedirs(tensorboard_path)

      output_img_path = args.output_img_path or 'results/anaoas'
      if os.path.exists(output_img_path):
        shutil.rmtree(output_img_path)
      os.makedirs(output_img_path)

      model.build()
      model.train(
          content_img_path=args.content_img_path or 'images/content/content1.jpg',
          style_img_path=args.style_img_path or 'images/style/style1.jpg',
          output_img_path=output_img_path,
          tensorboard_path=tensorboard_path)
    elif args.grid_search:
      # alfa_v = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,
                # 100, 300, 1000]
      # beta_v = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,
                # 100, 300, 1000]
      # learning_rate_v = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,
                       # 100, 300, 1000]
      alfa_v = [0.0001]
      beta_v = [0.1]
      learning_rate_v = [3]
      num_iters = 2000
      for learning_rate in learning_rate_v:
        for alfa in alfa_v:
          for beta in beta_v:
            tensorboard_path = args.tensorboard_path or 'tensorboard/tensorboard_anaoas'
            tensorboard_path = tensorboard_path \
                               + '_learning_rate_' + str(learning_rate) \
                               + '_alfa_' + str(alfa) \
                               + '_beta_' + str(beta)
            output_img_path = args.output_img_path or 'results/anaoas'
            output_img_path = output_img_path \
                              + '_learning_rate_' + str(learning_rate) \
                              + '_alfa_' + str(alfa) \
                              + '_beta_' + str(beta)
            if os.path.exists(tensorboard_path):
              shutil.rmtree(tensorboard_path)
            os.makedirs(tensorboard_path)
            if os.path.exists(output_img_path):
              shutil.rmtree(output_img_path)
            os.makedirs(output_img_path)

            content_img_path = args.content_img_path or 'images/content/content1.jpg'
            style_img_path = args.style_img_path or 'images/style/style1.jpg'
            content_img = cv2.imread(content_img_path)
            style_img = cv2.imread(style_img_path)

            model = anaoas.StyleTransfer(
                model_name=args.model_name or 'vgg19',
                tensorflow_model_path=args.tensorflow_model_path
                    or 'pretrained_models/vgg19/model/tensorflow/conv_wb.pkl',
                content_img_height=content_img.shape[0],
                content_img_width=content_img.shape[1],
                content_img_channels=content_img.shape[2],
                style_img_height=style_img.shape[0],
                style_img_width=style_img.shape[1],
                style_img_channels=style_img.shape[2],
                noise_img_height=content_img.shape[0],
                noise_img_width=content_img.shape[1],
                noise_img_channels=content_img.shape[2],
                content_layers=args.content_layers or ['conv4_2'],
                style_layers=args.style_layers
                    or ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                style_layers_w=args.style_layers_w
                    or [1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
                alfa=alfa,
                beta=beta,
                learning_rate=learning_rate,
                num_iters=num_iters)
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
    pass
  else:
    print('Nothing to be done!')

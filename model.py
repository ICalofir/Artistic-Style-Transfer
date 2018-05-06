import caffe
import pickle
import numpy as np

class Model():
  def __init__(self,
      caffe_model_path=
        'pretrained_models/vgg19/model/caffe/VGG_ILSVRC_19_layers.caffemodel',
      caffe_prototxt_path=
        'pretrained_models/vgg19/model/caffe/layer_configuration.prototxt',
      tensorflow_model_path=
        'pretrained_models/vgg19/model/tensorflow/conv_wb.pkl'):

    self.caffe_model_path = caffe_model_path
    self.caffe_prototxt_path = caffe_prototxt_path
    self.tensorflow_model_path = tensorflow_model_path

    self.net = caffe.Net(self.caffe_prototxt_path,
                         self.caffe_model_path,
                         caffe.TEST);

  def vgg19_caffe_to_tensorflow(self):
    layers = {}
    for layer_idx in range(len(self.net.layers)):
      if (self.net.layers[layer_idx].type.lower() != 'convolution'):
        continue

      layer = {}
      layer_name = self.net._layer_names[layer_idx]
      W_init = self.net.layers[layer_idx].blobs[0].data
      b = self.net.layers[layer_idx].blobs[1].data

      # W_init.shape = [num_outputs, num_inputs, k_h, k_w]
      # W.shape = [k_h, k_w, num_inputs, num_outputs]
      W = np.transpose(W_init, (2, 3, 1, 0))

      layer['weights'] = W
      layer['bias'] = b

      layers[layer_name] = layer

    with open(self.tensorflow_model_path, 'wb') as f:
      pickle.dump(layers, f)

if __name__ == '__main__':
  model = Model()
  model.vgg19_caffe_to_tensorflow()

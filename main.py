import a_neural_algorithm_of_artistic_style.style_transfer as anaoas
import perceptual_losses_for_real_time_style_transfer.style_transfer as plfrtst
import deep_photo_style_transfer.style_transfer as dpst

model_name = 'deep'

if __name__ == '__main__':
  if model_name == 'leon':
    model = anaoas.StyleTransfer(content_layers=['conv4_2'],
                          style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                          style_layers_w=[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
                          alfa=40,
                          beta=50,
                          learning_rate=10)
  elif model_name == 'fast':
    model = plfrtst.StyleTransfer(content_layers=['conv4_2'],
                          style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                          style_layers_w=[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
                          alfa=257,
                          beta=1,
                          learning_rate=0.001)
  elif model_name == 'deep':
    model = dpst.StyleTransfer(content_layers=['conv4_2'],
                          style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                          style_layers_w=[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
                          alfa=100,
                          beta=4,
                          gamma=100,
                          learning_rate=10)

  model.build()
  model.train()

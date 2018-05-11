import a_neural_algorithm_of_artistic_style.style_transfer as anaoas
import perceptual_losses_for_real_time_style_transfer.style_transfer as plfrtst

model_name = 'fast'

if __name__ == '__main__':
  if model_name == 'leon':
    model = anaoas.StyleTransfer(content_layers=['conv4_2'],
                          style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                          style_layers_w=[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
                          alfa=500,
                          beta=1)
  elif model_name == 'fast':
    model = plfrtst.StyleTransfer(content_layers=['conv4_2'],
                          style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                          style_layers_w=[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
                          alfa=500,
                          beta=1)

  model.build()
  model.train()

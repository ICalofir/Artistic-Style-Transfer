from style_transfer import StyleTransfer

if __name__ == '__main__':
  model = StyleTransfer(content_layers=['conv4_2'],
                        style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                        style_layers_w=[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0],
                        alfa=500,
                        beta=1)
  model.build()
  model.train()

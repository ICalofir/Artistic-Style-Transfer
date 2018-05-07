from style_transfer import StyleTransfer

if __name__ == '__main__':
  model = StyleTransfer()
  model.build()
  model.train()

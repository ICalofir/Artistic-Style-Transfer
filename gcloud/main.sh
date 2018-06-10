DIRECTORY="trainer"
if [ -d "$DIRECTORY" ]; then
  rm -rf $DIRECTORY
fi
mkdir $DIRECTORY

cp ../main.py $DIRECTORY/main.py
cp ../utils.py $DIRECTORY/utils.py
cp ../pretrained_models/model.py $DIRECTORY/model.py
cp ../a_neural_algorithm_of_artistic_style/anaoas_style_transfer.py $DIRECTORY/anaoas_style_transfer.py
cp ../perceptual_losses_for_real_time_style_transfer/plfrtst_style_transfer.py $DIRECTORY/plfrtst_style_transfer.py
cp ../perceptual_losses_for_real_time_style_transfer/dataset.py $DIRECTORY/dataset.py
cp ../deep_photo_style_transfer/dpst_style_transfer.py $DIRECTORY/dpst_style_transfer.py
cp ../conv_nets/transform_net.py $DIRECTORY/transform_net.py
cp ../conv_nets/vgg19.py $DIRECTORY/vgg19.py
touch $DIRECTORY/__init__.py

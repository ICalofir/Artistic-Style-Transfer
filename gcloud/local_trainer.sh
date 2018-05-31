TENSORFLOW_MODEL_PATH=../../pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
CONTENT_IMG_PATH=../../images/content/content1.jpg
STYLE_IMG_PATH=../../images/style/style1.jpg
OUTPUT_IMG_PATH=../../results/anaoas/local
TENSORBOARD_PATH=../../tensorboard/tensorboard_anaoas/testt
METHOD_NAME=anaoas

python main.py \
    --method $METHOD_NAME \
    --train \
    --tensorflow_model_path $TENSORFLOW_MODEL_PATH \
    --content_img_path $CONTENT_IMG_PATH \
    --style_img_path $STYLE_IMG_PATH \
    --output_img_path $OUTPUT_IMG_PATH \
    --tensorboard_path $TENSORBOARD_PATH

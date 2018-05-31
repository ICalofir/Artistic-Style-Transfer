METHOD_NAME=anaoas
TENSORFLOW_MODEL_PATH=../../pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
CONTENT_IMG_SIZE=$7
STYLE_IMG_SIZE=$8
ALFA=$1
BETA=$2
LEARNING_RATE=$3
NUM_ITERS=$4
CONTENT_IMG_PATH=../../images/content/$5
STYLE_IMG_PATH=../../images/style/$6
OUTPUT_IMG_PATH=../../results/anaoas/alfa_${ALFA}_beta_${BETA}_lr_${LEARNING_RATE}
TENSORBOARD_PATH=../../tensorboard/tensorboard_anaoas/alfa_${ALFA}_beta_${BETA}_lr_${LEARNING_RATE}

python main.py \
    --method $METHOD_NAME \
    --train \
    --tensorflow_model_path $TENSORFLOW_MODEL_PATH \
    --alfa $ALFA \
    --beta $BETA \
    --learning_rate $LEARNING_RATE \
    --num_iters $NUM_ITERS \
    --content_img_size $CONTENT_IMG_SIZE \
    --style_img_size $STYLE_IMG_SIZE \
    --content_img_path $CONTENT_IMG_PATH \
    --style_img_path $STYLE_IMG_PATH \
    --output_img_path $OUTPUT_IMG_PATH \
    --tensorboard_path $TENSORBOARD_PATH

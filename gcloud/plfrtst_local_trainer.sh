METHOD_NAME=plfrtst
TENSORFLOW_MODEL_PATH=../../pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
DATA_PATH=../../perceptual_losses_for_real_time_style_transfer/dataset
ALFA=$1
BETA=$2
GAMMA=$3
LEARNING_RATE=$4
BATCH_SIZE=$5
NO_EPOCHS=$6
STYLE_IMG_PATH=../../images/style/$7
OUTPUT_IMG_PATH=../../results/plfrtst/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_lr_${LEARNING_RATE}
TENSORBOARD_PATH=../../tensorboard/tensorboard_plfrtst/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_lr_${LEARNING_RATE}
MODEL_PATH=../../models/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_lr_${LEARNING_RATE}

python main.py \
    --method $METHOD_NAME \
    --train \
    --tensorflow_model_path $TENSORFLOW_MODEL_PATH \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --alfa $ALFA \
    --beta $BETA \
    --gamma $GAMMA \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --no_epochs $NO_EPOCHS \
    --style_img_path $STYLE_IMG_PATH \
    --output_img_path $OUTPUT_IMG_PATH \
    --tensorboard_path $TENSORBOARD_PATH

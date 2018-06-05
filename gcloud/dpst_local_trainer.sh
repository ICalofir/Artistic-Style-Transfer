METHOD_NAME=dpst
TENSORFLOW_MODEL_PATH=../../pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
CONTENT_IMG_SIZE=${13}
STYLE_IMG_SIZE=${14}
ALFA=$1
BETA=$2
GAMMA=$3
LLAMBDA=$4
LEARNING_RATE=$5
NUM_ITERS=$6
CONTENT_IMG_PATH=../../images/content/$7
STYLE_IMG_PATH=../../images/style/$8
NOISE_IMG_PATH=../../images/content/$9
OUTPUT_IMG_PATH=../../results/dpst/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_llambda_${LLAMBDA}_lr_${LEARNING_RATE}
TENSORBOARD_PATH=../../tensorboard/tensorboard_dpst/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_llambda_${LLAMBDA}_lr_${LEARNING_RATE}
MASK_CONTENT_IMG_PATH=../../images/mask/${10}
MASK_STYLE_IMG_PATH=../../images/mask/${11}
LAPLACIAN_MATRIX_PATH=../../images/laplacian/${12}

python main.py \
    --method $METHOD_NAME \
    --train \
    --tensorflow_model_path $TENSORFLOW_MODEL_PATH \
    --alfa $ALFA \
    --beta $BETA \
    --gamma $GAMMA \
    --llambda $LLAMBDA \
    --learning_rate $LEARNING_RATE \
    --num_iters $NUM_ITERS \
    --content_img_size $CONTENT_IMG_SIZE \
    --style_img_size $STYLE_IMG_SIZE \
    --content_img_path $CONTENT_IMG_PATH \
    --style_img_path $STYLE_IMG_PATH \
    --noise_img_path $NOISE_IMG_PATH\
    --output_img_path $OUTPUT_IMG_PATH \
    --tensorboard_path $TENSORBOARD_PATH \
    --mask_content_img_path $MASK_CONTENT_IMG_PATH \
    --mask_style_img_path $MASK_STYLE_IMG_PATH \
    --laplacian_matrix_path $LAPLACIAN_MATRIX_PATH

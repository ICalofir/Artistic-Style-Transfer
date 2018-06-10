PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=europe-west1
JOB_NAME=$1
STAGING_BUCKET=gs://$BUCKET_NAME

METHOD_NAME=plfrtst
TENSORFLOW_MODEL_PATH=gs://$BUCKET_NAME/pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
DATA_PATH=gs://$BUCKET_NAME/perceptual_losses_for_real_time_style_transfer/dataset
ALFA=$2
BETA=$3
GAMMA=$4
LEARNING_RATE=$5
BATCH_SIZE=$6
NO_EPOCHS=$7
CONTENT_IMG_PATH=gs://$BUCKET_NAME/images/content/$8
STYLE_IMG_PATH=gs://$BUCKET_NAME/images/style/$9
OUTPUT_IMG_PATH=gs://$BUCKET_NAME/results/plfrtst/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_lr_${LEARNING_RATE}
TENSORBOARD_PATH=gs://$BUCKET_NAME/tensorboard/tensorboard_plfrtst/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_lr_${LEARNING_RATE}
MODEL_PATH=gs://$BUCKET_NAME/models/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_lr_${LEARNING_RATE}
CHECKPOINTS_PATH=gs://$BUCKET_NAME/checkpoints/alfa_${ALFA}_beta_${BETA}_gamma_${GAMMA}_lr_${LEARNING_RATE}

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket=${STAGING_BUCKET} \
    --runtime-version 1.7 \
    --module-name trainer.main \
    --package-path trainer/ \
    --region $REGION \
    --config config.yaml \
    -- \
    --method $METHOD_NAME \
    --train \
    --tensorflow_model_path $TENSORFLOW_MODEL_PATH \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --alfa $ALFA \
    --beta $BETA \
    --gamma $GAMMA \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --no_epochs $NO_EPOCHS \
    --content_img_path $CONTENT_IMG_PATH \
    --style_img_path $STYLE_IMG_PATH \
    --output_img_path $OUTPUT_IMG_PATH \
    --tensorboard_path $TENSORBOARD_PATH \
    --checkpoints_path $CHECKPOINTS_PATH

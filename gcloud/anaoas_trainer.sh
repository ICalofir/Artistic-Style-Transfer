PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=us-central1
JOB_NAME=$1
STAGING_BUCKET=gs://$BUCKET_NAME

METHOD_NAME=anaoas
TENSORFLOW_MODEL_PATH=gs://$BUCKET_NAME/pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
CONTENT_IMG_SIZE=$8
STYLE_IMG_SIZE=$9
ALFA=$2
BETA=$3
LEARNING_RATE=$4
NUM_ITERS=$5
CONTENT_IMG_PATH=gs://$BUCKET_NAME/images/content/$6
STYLE_IMG_PATH=gs://$BUCKET_NAME/images/style/$7
OUTPUT_IMG_PATH=gs://$BUCKET_NAME/results/anaoas/alfa_${ALFA}_beta_${BETA}_lr_${LEARNING_RATE}
TENSORBOARD_PATH=gs://$BUCKET_NAME/tensorboard/tensorboard_anaoas/alfa_${ALFA}_beta_${BETA}_lr_${LEARNING_RATE}

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

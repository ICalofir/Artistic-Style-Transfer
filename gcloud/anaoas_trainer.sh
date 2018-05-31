PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=us-central1
JOB_NAME=$1
STAGING_BUCKET=gs://$BUCKET_NAME

TENSORFLOW_MODEL_PATH=gs://$BUCKET_NAME/pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
CONTENT_IMG_PATH=gs://$BUCKET_NAME/images/content/content1.jpg
STYLE_IMG_PATH=gs://$BUCKET_NAME/images/style/style1.jpg
OUTPUT_IMG_PATH=gs://$BUCKET_NAME/results/anaoas/testt
TENSORBOARD_PATH=gs://$BUCKET_NAME/tensorboard/tensorboard_anaoas/testt
METHOD_NAME=anaoas

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket=${STAGING_BUCKET} \
    --runtime-version 1.7 \
    --module-name trainer.main \
    --package-path trainer/ \
    --region $REGION \
    --config config.yaml \
    -- \
    --method $METHOD_NAME \
    --grid_search \
    --tensorflow_model_path $TENSORFLOW_MODEL_PATH \
    --content_img_path $CONTENT_IMG_PATH \
    --style_img_path $STYLE_IMG_PATH \
    --output_img_path $OUTPUT_IMG_PATH \
    --tensorboard_path $TENSORBOARD_PATH

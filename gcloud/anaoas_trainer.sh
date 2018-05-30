PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=us-central1
JOB_NAME=$1
STAGING_BUCKET=gs://$BUCKET_NAME

TENSORFLOW_MODEL_PATH=gs://$BUCKET_NAME/pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
CONTENT_IMG_PATH=gs://$BUCKET_NAME/images/content/content1.jpg
STYLE_IMG_PATH=gs://$BUCKET_NAME/images/style/style1.jpg
OUTPUT_IMG_PATH=gs://$BUCKET_NAME/results/anaoas
TENSORBOARD_PATH=gs://$BUCKET_NAME/tensorboard/tensorboard_anaoas
METHOD_NAME=anaoas

L_TENSORFLOW_MODEL_PATH=../pretrained_models/vgg19/model/tensorflow/conv_wb.pkl
L_CONTENT_IMG_PATH=../images/content/content1_resized.jpg
L_STYLE_IMG_PATH=../images/style/style1_resized.jpg
L_OUTPUT_IMG_PATH=../results/anaoas
L_TENSORBOARD_PATH=..//tensorboard/tensorboard_anaoas
L_METHOD_NAME=anaoas

CPATH=gs://$BUCKET_NAME/images/content/content1

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

#gcloud ml-engine local train \
    #--module-name trainer.main \
    #--package-path trainer/ \
    #-- \
    #--method $L_METHOD_NAME \
    #--grid_search \
    #--tensorflow_model_path $L_TENSORFLOW_MODEL_PATH \
    #--content_img_path $L_CONTENT_IMG_PATH \
    #--style_img_path $L_STYLE_IMG_PATH \
    #--output_img_path $L_OUTPUT_IMG_PATH \
    #--tensorboard_path $L_TENSORBOARD_PATH

#gcloud ml-engine jobs submit training $JOB_NAME \
    #--staging-bucket=${STAGING_BUCKET} \
    #--module-name testt.main \
    #--package-path testt/ \
    #--region $REGION \
    #-- \
    #--path $CPATH

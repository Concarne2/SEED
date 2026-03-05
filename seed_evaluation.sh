RECON_IMAGE_PATH='path/to/recon/images'
GT_IMAGE_PATH='path/to/gt/images'
CONFIG_PATH='mmdet/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py'
WEIGHTS_PATH='grounding_dino_swin-l_pretrain_all-56d69e78.pth'
DECODING_MODEL_NAME='test_model'

python image_detection.py $RECON_IMAGE_PATH \
        $CONFIG_PATH \
        --weights $WEIGHTS_PATH \
        --texts '$: coco_gender' \
        --out-dir "evaluations/$DECODING_MODEL_NAME/recon_detection_results" 

wait

python image_detection.py $GT_IMAGE_PATH \
        $CONFIG_PATH \
        --weights $WEIGHTS_PATH \
        --texts '$: coco_gender' \
        --out-dir "evaluations/$DECODING_MODEL_NAME/gt_detection_results" 
wait

python recon_evaluation.py \
        --recon-detection-dir "evaluations/$DECODING_MODEL_NAME/recon_detection_results" \
        --gt-detection-dir "evaluations/$DECODING_MODEL_NAME/gt_detection_results" \
        --recon-img-dir $RECON_IMAGE_PATH \
        --gt-img-dir $GT_IMAGE_PATH \
        --model-name $DECODING_MODEL_NAME \
        --intermediate-results-dir "evaluations/$DECODING_MODEL_NAME/intermediate_results" 

# add optional wandb arguments if needed
# --wandb-project "your-project-name" \
# --wandb-run-name "your-run-name" \
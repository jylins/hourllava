# ignore warnings
export PYTHONWARNINGS="ignore"

export HF_TOKEN="YOUR_HF_TOKEN"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

DATA_PATH="./experiments/streaming/hourllava_s1.yaml"
IMAGE_FOLDER="/path/to/image_folder"
VIDEO_FOLDER="/path/to/video_folder"

PROMPT_VERSION="qwen_1_5"

########### Stage 1: Image-Language Pretraining ###########
RUN_NAME="hourllava-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-s1" 
PREV_STAGE_CHECKPOINT="jylins/llava-onevision-qwen25-3b-si"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"

NUM_GPUS=8
NNODES=8
RANK=$1
ADDR="YOUR_ADDR"
PORT=30000

mkdir -p logs/"$RUN_NAME"


HSA_FORCE_FINE_GRAIN_PCIE=1 OMP_NUM_THREADS=16 NCCL_DEBUG=INFO torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --master_addr="${ADDR}" --node_rank="${RANK}" --master_port="${PORT}" \
    hourllava/train/train.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --dispatch_batches False \
    --use_streaming True \
    --using_local_data True \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --video_folder ${VIDEO_FOLDER} \
    --mm_tunable_parts="mm_vision_resampler" \
    --mm_resampler_lr=1e-4 \
    --mm_projector_type mlp2x_gelu \
    --mm_resampler_type memaug \
    --memaug_depth 4 \
    --spatial_sampling_type random \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./work_dirs/hourllava/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --batching_method "random" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32
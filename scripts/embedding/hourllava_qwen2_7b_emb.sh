LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL=./work_dirs/hourllava-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-s2/vision_tower.bin

mkdir -p ./logs

for GPU in {0..7}
do
SPLITS=8
echo "CUDA_VISIBLE_DEVICES=${GPU} python scripts/embedding/videomarathon_emb.py --idx $1 --splits ${SPLITS} --min_frames 128 --vision_tower ${VISION_MODEL}"
CUDA_VISIBLE_DEVICES=${GPU} nohup python scripts/embedding/videomarathon_emb.py \
    --idx $1 \
    --splits ${SPLITS} \
    --min_frames 128 \
    --vision_tower ${VISION_MODEL} \
    > ./logs/output_emb_hourllava-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-s2_GPU${GPU}.log 2>&1 &
done
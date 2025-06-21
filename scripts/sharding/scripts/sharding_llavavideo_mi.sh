mkdir -p ./data/VideoMarathon/sharding

for i in {0..9}
do
    python scripts/sharding/sharding_multi_image_data.py \
        --image_folder images \
        --yaml_file ./experiments/llavavideo_mi.yaml \
        --root_path ./data/LLaVA-OneVision \
        --output_folder ./data/VideoMarathon/sharding/llavavideo_mi_part${i} \
        --processes 64 \
        --shard_id $i \
        --num_shards 10
done

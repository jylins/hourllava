mkdir -p ./data/VideoMarathon/sharding

for i in {0..9}
do
    python scripts/sharding/sharding_single_image_data.py \
        --image_folder images \
        --yaml_file ./experiments/llava_ov_si.yaml \
        --root_path ./data/LLaVA-OneVision \
        --output_folder ./data/VideoMarathon/sharding/llava_ov_si_onlyimg_part${i} \
        --processes 128 \
        --shard_id $i \
        --exclude_text 1 \
        --num_shards 10
done

mkdir -p ./data/VideoMarathon/sharding

for i in {0..9}
do
    python scripts/sharding/sharding_video_str_data.py \
        --yaml_file ./experiments/llavavideo_video.yaml \
        --root_path ./data/LLaVA-Video-178K/videos \
        --output_folder ./data/VideoMarathon/sharding/llavavideo_video_part${i} \
        --processes 64 \
        --shard_id $i \
        --num_shards 10
done

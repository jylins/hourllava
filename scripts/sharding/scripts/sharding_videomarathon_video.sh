mkdir -p ./data/VideoMarathon/sharding

for i in {0..9}
do
    python scripts/sharding/sharding_videofeat_str_data.py \
        --yaml_file ./experiments/videomarathon.yaml \
        --root_path ./data/VideoMarathon/videos \
        --output_folder ./data/VideoMarathon/sharding/videomarathon_video_part${i} \
        --processes 64 \
        --shard_id $i \
        --num_shards 10
done

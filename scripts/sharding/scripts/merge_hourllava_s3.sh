OUTPUT_DIR=$(realpath ./data/VideoMarathon)

mkdir -p ${OUTPUT_DIR}/sharding/hourllava_s3/0
mkdir -p ${OUTPUT_DIR}/sharding/hourllava_s3/1

mkdir -p ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_si_part5
mkdir -p ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_mi_part5
mkdir -p ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_text_part5
mkdir -p ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_si_part5
mkdir -p ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_mi_part5
mkdir -p ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_text_part5


###############################################################
# LLaVA-Video (short videos) + half LLaVA-OV (si + mi + text)
###############################################################
for i in {2..9}
do
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_video_part${i} ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_video_part${i}
done

for i in {1..4}
do
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_text_part${i} ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_text_part${i}
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_si_part${i} ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_si_part${i}
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_mi_part${i} ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_mi_part${i}
done

for i in {0..7}
do
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_text_part5/${i} ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_text_part5/${i}
done

for i in {0..31}
do
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_si_part5/${i} ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_si_part5/${i}
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_mi_part5/${i} ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_mi_part5/${i}
done

###############################################################
# VideoMarathon (long videos) + half LLaVA-OV (si + mi + text)
###############################################################
for i in {0..9}
do
    ln -s ${OUTPUT_DIR}/sharding/videomarathon_video_part${i} ${OUTPUT_DIR}/sharding/hourllava_s3/1/videomarathon_video_part${i}
done

for i in {6..9}
do
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_text_part${i} ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_text_part${i}
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_si_part${i} ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_si_part${i}
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_mi_part${i} ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_mi_part${i}
done

for i in {8..15}
do
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_text_part5/${i} ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_text_part5/${i}
done

for i in {32..63}
do
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_si_part5/${i} ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_si_part5/${i}
    ln -s ${OUTPUT_DIR}/sharding/llavavideo_mi_part5/${i} ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_mi_part5/${i}
done

###############################################################
# merge shards
###############################################################
python scripts/sharding/merge_shards.py \
    --root ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_text_part5
python scripts/sharding/merge_shards.py \
    --root ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_si_part5
python scripts/sharding/merge_shards.py \
    --root ${OUTPUT_DIR}/sharding/hourllava_s3/0/llavavideo_mi_part5

python scripts/sharding/merge_shards.py \
    --root ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_text_part5
python scripts/sharding/merge_shards.py \
    --root ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_si_part5
python scripts/sharding/merge_shards.py \
    --root ${OUTPUT_DIR}/sharding/hourllava_s3/1/llavavideo_mi_part5

python scripts/sharding/merge_shards.py \
    --root ${OUTPUT_DIR}/sharding/hourllava_s3/0
python scripts/sharding/merge_shards.py \
    --root ${OUTPUT_DIR}/sharding/hourllava_s3/1
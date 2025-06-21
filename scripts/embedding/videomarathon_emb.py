from glob import glob
import argparse

from transformers import AutoModel
import torch

from hourllava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

import os
from decord import VideoReader, cpu
import numpy as np
import torch.nn as nn
import math
from tqdm import tqdm
from glob import glob
import av


def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames


def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def read_video_pyav(video_path, num_frm=128):
    container = av.open(video_path)

    if "webm" not in video_path and "mkv" not in video_path:
        total_frames = container.streams.video[0].frames
        fps = float(container.streams.video[0].average_rate)
        duration = int(total_frames / fps)
        sampled_frm = max(duration, num_frm)
        if num_frm > total_frames:
            sampled_frm = total_frames
            indices = np.arange(total_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
        # Append the last frame index if not already included
        if total_frames - 1 not in indices:
            indices = np.append(indices, total_frames - 1)
        frames = record_video_length_stream(container, indices)
    else:
        frames = record_video_length_packet(container)
        fps = float(container.streams.video[0].average_rate)
        total_frames = len(frames)
        duration = int(total_frames / fps)
        sampled_frm = max(duration, num_frm)
        if num_frm > total_frames:
            sampled_frm = total_frames
            indices = np.arange(total_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
        # Append the last frame index if not already included
        if total_frames - 1 not in indices:
            indices = np.append(indices, total_frames - 1)
        frames = [frames[i] for i in indices]
    return sampled_frm, round(float(total_frames / fps), 2), np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_2dPool(image_feature, model, stride=2, scaled_shape=None):
    height = width = model.num_patches_per_side
    num_frames, num_tokens, num_dim = image_feature.shape
    image_feature = image_feature.view(num_frames, height, width, -1)
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
    # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
    if scaled_shape is None:
        height, width = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
    image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
    image_feature = image_feature.permute(0, 2, 3, 1)
    image_feature = image_feature.view(num_frames, -1, num_dim)
    return image_feature


def main(args):
    root = './data/VideoMarathon'
    videos = glob(f'{root}/sub_videos/*/*.mp4', recursive=True)
    print(f'Number of videos: {len(videos)}')

    videos = sorted(videos)
    videos = videos[args.idx::args.splits]
    print(f'Number of sampled videos: {len(videos)}')

    # load video model
    model = SigLipVisionTower("google/siglip-so400m-patch14-384")
    keys = model.load_state_dict(torch.load(args.vision_tower), strict=True)
    print(keys)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for video_pth in tqdm(videos):
            # load video data
            output_file = video_pth.replace("VideoMarathon/videos", "VideoMarathon/features").replace(".mp4", ".pt")
            if os.path.exists(output_file):
                print(f'Skipped {output_file}')
                continue
            sampled_frm, duration, video = read_video_pyav(video_pth, num_frm=args.min_frames)
            processor = model.image_processor
            image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
            image = image.to('cuda')
            
            # encode video
            if image.shape[0] > 128:
                # chunked video frames
                image_features = []
                concat_images_chunked = image.split(128, dim=0)
                for chunked_images in concat_images_chunked:
                    tmp_features = model(chunked_images)
                    image_features.append(tmp_features.detach().cpu())
                    del tmp_features
                    torch.cuda.empty_cache()
                encoded_image_features = torch.cat(image_features, dim=0).to(image.device)
            else:
                encoded_image_features = model(image)
            
            # pooling
            vid_feat = get_2dPool(encoded_image_features, model=model, scaled_shape=[8, 8]).detach().cpu()
            feature = {
                "vid_feat": vid_feat,
                "duration": round(float(duration), 2),
            }
            
            # save features
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            torch.save(feature, output_file)
            print(f'Saved {output_file}: vid_feat={vid_feat.shape}, duration={round(float(duration), 2)}')
            del image, encoded_image_features, vid_feat, feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--splits", type=int, default=128)
    parser.add_argument("--min_frames", type=int, default=128)
    parser.add_argument("--vision_tower", type=str, default="/path/to/vision_tower.bin")
    args = parser.parse_args()
    main(args)
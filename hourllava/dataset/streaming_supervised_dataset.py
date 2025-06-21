import os
import yaml
import time
import copy
from typing import Dict, Optional
from PIL import Image
import torch
import transformers
import numpy as np
from decord import VideoReader, cpu

import streaming
import math

from hourllava.util import DataArguments
from hourllava.util.constants import DEFAULT_IMAGE_TOKEN
from hourllava.util import process_highres_image, process_anyres_image, process_highres_image_crop_split

from .preprocess import preprocess, preprocess_multimodal


class StreamingSupervisedDataset(streaming.StreamingDataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 local: str = None,
                 shuffle: bool = True,
                 predownload: Optional[int] = 100_000,
                 download_retry: Optional[int] = 2,
                 download_timeout: Optional[float] = 120,
                 batch_size: Optional[int] = None, 
                 batching_method: Optional[str] = 'per_stream',
                 **kwargs) -> None:
        
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        
        datasets = yaml_data.get("datasets")

        streams = []  # support mixture of shards, see https://github.com/mosaicml/streaming/blob/main/docs/source/dataset_configuration/mixing_data_sources.md
        for idx, dataset in enumerate(datasets):
            if dataset.get('choose', None):
                tmp_stream = streaming.Stream(
                    remote=dataset.get("shard_path"),
                    local = os.path.join(local, f'tmp_{idx}'),
                    choose = dataset.get('choose')
                )
            elif dataset.get('local_path', None):
                tmp_stream = streaming.Stream(
                    local = dataset.get('local_path'),
                )
            else:
                tmp_stream = streaming.Stream(
                    remote = dataset.get("shard_path"),
                    local = os.path.join(local, f'tmp_{idx}')
                )
            streams.append(tmp_stream)
        super().__init__(streams=streams,
                         shuffle=shuffle,
                         predownload=predownload,
                         keep_zip=False,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=None,
                         batch_size=batch_size,
                         batching_method=batching_method,
                         **kwargs)

        self.tokenizer = tokenizer
        self.data_args = data_args

    def process_image(self, image, overwrite_image_aspect_ratio=None):
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, self.num_samples - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        _sources = self.get_item(i)
        if "image" in _sources and type(_sources["image"]) is not list:
            if _sources["image"].size == (1, 1):
                _sources.pop("image")
        if "text" in _sources:
            _sources["conversations"] = _sources["text"]["conversations"]
            if "id" in _sources["text"]:
                _sources["id"] = _sources["text"]["id"]
            _sources.pop("text")
        sources = [_sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = _sources["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sources[0]:
            # Note: the mds file only contains the path to the video file, rather the video bytes
            video_file = _sources["video"]
            if video_file.endswith(".pt"):
                video = torch.load(video_file)
                video_time = video.pop('duration')
                image = video['vid_feat']
                sampled_frm = min(image.shape[0], int(video_time))
                uniform_sampled_frames = np.linspace(0, image.shape[0] - 1, sampled_frm, dtype=int)
                image = image[uniform_sampled_frames]
                side = int(math.sqrt(image.shape[1]))
                image = image.reshape(-1, side, side, image.shape[-1])
                num_frames_to_sample = image.shape[0]
                
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, None, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            else:
                vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                
                total_frame_num = len(vr)
                video_time = total_frame_num / vr.get_avg_fps()
                avg_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
                frame_time = [i/avg_fps for i in frame_idx]
                
                if len(frame_idx) < self.data_args.min_frames:
                    sampled_frm = min(total_frame_num, self.data_args.min_frames)
                    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sampled_frm, dtype=int)
                    frame_idx = uniform_sampled_frames.tolist()
                    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
                
                if self.data_args.frames_upbound > 0:
                    if len(frame_idx) > self.data_args.frames_upbound or self.data_args.force_sample:
                        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.data_args.frames_upbound, dtype=int)
                        frame_idx = uniform_sampled_frames.tolist()
                        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
                
                video = vr.get_batch(frame_idx).asnumpy()
                frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                num_frames_to_sample = num_frames = len(frame_idx)
                # https://github.com/dmlc/decord/issues/208
                vr.seek(0)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in _sources) or ("video" in _sources)
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in _sources:
            data_dict["image"] = image
        elif "video" in _sources:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = _sources.get("id", i)

        return data_dict
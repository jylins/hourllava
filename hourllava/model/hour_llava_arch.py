#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from einops import rearrange

import random
import numpy as np
import math
import re
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from hourllava.util.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from hourllava.util import rank0_print, get_anyres_image_grid_shape
import torch.distributed as dist
import gc

import time
import torch.nn.functional as F


class HourLlavaMetaModel:

    def __init__(self, config):
        super(HourLlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            self.vision_resampler = build_vision_resampler(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")
        
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(self.config)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_tower, "hidden_size", 1152)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        self.spatial_sampling_type = getattr(self.config, "spatial_sampling_type", 'random')
        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)
            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
            
            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"), strict=False)
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class HourLlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2, scaled_shape=None):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            if scaled_shape is None:
                height, width = image_feature.shape[2:]
                scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_offload(self, images):
        with torch.no_grad():
            image_features = []
            # chunked concat images
            concat_images_chunked = images.split(128, dim=0)
            for chunked_images in concat_images_chunked:
                tmp_features = self.get_model().get_vision_tower()(chunked_images)
                image_features.append(tmp_features.detach().cpu())
                del tmp_features
                torch.cuda.empty_cache()
            image_features = torch.cat(image_features, dim=0).to(images.device)
        # image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature
    
    def multi_image_resize(self, image_feature, image_size, image_aspect_ratio, mm_patch_merge_type, num_patches_per_side):
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]
        height = width = num_patches_per_side
        assert height * width == base_image_feature.shape[0]
        base_image_feature = base_image_feature.view(height, width, -1)

        if "anyres_max" in image_aspect_ratio:
            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
            if matched_anyres_max_num_patches:
                max_num_patches = int(matched_anyres_max_num_patches.group(1))

        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            if hasattr(self.get_vision_tower(), "image_size"):
                vision_tower_image_size = self.get_vision_tower().image_size
            else:
                raise ValueError("vision_tower_image_size is not found in the vision tower.")
            try:
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_size, self.config.image_grid_pinpoints, vision_tower_image_size)
            except Exception as e:
                rank0_print(f"Error: {e}")
                num_patch_width, num_patch_height = 2, 2
            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
        else:
            image_feature = image_feature.view(2, 2, height, width, -1)

        if "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
            unit = image_feature.shape[2]
            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
            image_feature = unpad_image(image_feature, image_size)
            c, h, w = image_feature.shape
            times = math.sqrt(h * w / (max_num_patches * unit**2))
            if times > 1.1:
                image_feature = image_feature[None]
                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
            image_feature = image_feature.permute(1, 2, 0).contiguous()
        elif "unpad" in mm_patch_merge_type:
            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
            image_feature = unpad_image(image_feature, image_size)
            image_feature = image_feature.permute(1, 2, 0).contiguous()
        else:
            raise NotImplementedError 
        return base_image_feature, image_feature
    
    def single_image_resize(self, image_feature, mm_patch_merge_type):
        image_feature = image_feature[0]
        if "unpad" in mm_patch_merge_type:
            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)
        return image_feature
    
    def get_modalities(self, modalities, input_ids):
        if isinstance(modalities, str):
            modalities = [modalities]
        num_images_list = []
        for cur_input_ids in input_ids:
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_images_list.append(num_images)
        assert len([modality for modality in modalities if modality != 'text']) == sum([num_images.item() for num_images in num_images_list])
        
        new_modalities = []
        image_idx = 0
        for idx, cur_input_ids in enumerate(input_ids):
            modality = modalities[image_idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            image_idx += num_images
            if num_images == 0:
                image_idx += 1
                new_modalities.append({"text": 1})
            elif num_images == 1:
                if modality == "image":
                    new_modalities.append({"image": 1})
                elif modality == "video":
                    new_modalities.append({"video": 1})
                else:
                    raise ValueError(f"Unexpected modality: {modality}")
            elif num_images > 1:
                new_modalities.append({"multi_image": num_images})
            else:
                raise ValueError(f"Unexpected num_images: {num_images}")
        return new_modalities
    
    def arange_images(self, images, image_sizes, modalities):
        assert type(images) is list
        images = [image.unsqueeze(0) if image.ndim == 3 else image for image in images]
        
        new_images = []
        new_image_sizes = []
        start_idx = 0
        end_idx = 0
        for modality in modalities:
            for _, value in modality.items():
                end_idx += value
                new_images.append(torch.cat(images[start_idx:end_idx], dim=0))
                new_image_sizes.append(image_sizes[start_idx:end_idx])
                start_idx += value
        return new_images, new_image_sizes
    
    def q_vision_sampling(self, kv_image_features, kv_position_ids_list, ratio=0.25):
        B, H, W, C = kv_image_features.shape

        if self.get_model().spatial_sampling_type == 'uniform':
            q_image_features = kv_image_features[:, ::2, ::2, :].reshape(B, -1, C)
            q_position_ids_list = kv_position_ids_list[:, ::2, ::2].reshape(B, -1)
        elif self.get_model().spatial_sampling_type == 'random':
            flat_N = H * W
            random_sampling_num = math.ceil(H * W * ratio)
            random_vals = torch.rand(B, flat_N)
            _, random_indices = torch.topk(random_vals, random_sampling_num, dim=1, largest=False)
            random_indices, _ = torch.sort(random_indices, dim=1)
            random_indices = random_indices.to(kv_image_features.device)
            q_image_features = torch.gather(
                kv_image_features.reshape(B, flat_N, C),
                1,
                random_indices.unsqueeze(-1).expand(-1, -1, C)
            )
            q_position_ids_list = torch.gather(
                kv_position_ids_list.reshape(B, flat_N),
                1,
                random_indices
            )
        else:
            raise ValueError(f"Unexpected spatial_sampling_type: {self.get_model().spatial_sampling_type}")

        return q_image_features, q_position_ids_list
    
    def add_grid_image_newline(self, image_feature, image_position_ids, add_positions, image_newline_token_id):
        image_newline_token = self.model.image_newline[None, :]
        _image_newline_token_id = torch.tensor([image_newline_token_id], device=image_position_ids.device)
        image_feature_segments = []
        position_ids_segments = []
        start_idx = 0
        for pos in add_positions:
            # find the position to insert the newline token
            insert_idx = (image_position_ids >= pos).nonzero(as_tuple=True)[0]
            if len(insert_idx) == 0:
                continue
            insert_idx = insert_idx[0].item()
            if image_position_ids[start_idx:insert_idx].shape[0] == 0:
                continue
            image_feature_segments.append(image_feature[start_idx:insert_idx])
            image_feature_segments.append(image_newline_token)
            position_ids_segments.append(image_position_ids[start_idx:insert_idx])
            position_ids_segments.append(_image_newline_token_id)
            start_idx = insert_idx
        
        image_feature = torch.cat(image_feature_segments + [image_feature[start_idx:], image_newline_token], dim=0)
        image_position_ids = torch.cat(position_ids_segments + [image_position_ids[start_idx:], _image_newline_token_id], dim=0)
        
        del image_feature_segments
        del position_ids_segments
        
        return image_feature, image_position_ids

    def local_avg_cosine(self, x: torch.Tensor, w: int = 9) -> torch.Tensor:
        """
        x: (T, D) tensor of embeddings
        w: window size (must be odd)
        returns: (T,) tensor of local average cosine similarities
        """
        assert w % 2 == 1, "window size must be odd"
        T, _ = x.shape
        r = w // 2

        x_norm = F.normalize(x, p=2, dim=1)                  # (T, D)
        S = x_norm @ x_norm.transpose(0, 1)                  # (T, T), S[t, t'] = cos(x[t], x[t'])
        S_pad = F.pad(S, (r, r))                             # (T, T + 2*r)
        windows = S_pad.unfold(dimension=1, size=w, step=1)  # (T, T, w)
        sum_windows = windows.sum(dim=2)                     # (T, T)
        local_sums = sum_windows.diagonal()                  # (T,)
        sums_excl_self = local_sums - 1.0
        # compute, for each t, how many neighbors actually exist:
        idx = torch.arange(T)
        left  = torch.clamp(idx - r, min=0)
        right = torch.clamp(idx + r, max=T - 1)
        neighbor_counts = (right - left + 1) - 1             # exclude self

        local_avg = sums_excl_self / neighbor_counts.to(x.dtype).to(x.device)

        return local_avg

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            # get modalities
            modalities = self.get_modalities(modalities, input_ids)
            
            images, image_sizes = self.arange_images(images, image_sizes, modalities)
            
            # vision encoding + mlp projection
            images_list = [img if img.ndim == 4 else img.unsqueeze(0) for img in images]
            split_sizes = [img.shape[0] for img in images_list]
            encoded_image_features = []
            for concat_images in images_list:
                if concat_images.shape[-1] == 1152:
                    encoded_image_features.append(concat_images.flatten(1, 2))
                else:
                    if concat_images.shape[0] > 128:
                        encoded_image_features.append(self.encode_images_offload(concat_images))
                    else:
                        encoded_image_features.append(self.encode_images(concat_images))
            
            new_encoded_image_features = []
            for idx, modality in enumerate(modalities):
                if 'video' in modality.keys():
                    if encoded_image_features[idx].shape[1] <= 64:
                        image_feat = self.get_model().mm_projector(encoded_image_features[idx])
                    else:
                        image_feat = self.get_model().mm_projector(self.get_2dPool(encoded_image_features[idx], scaled_shape=[8, 8]))
                    if image_feat.shape[0] < 128:
                        sampled_frames = np.linspace(0, image_feat.shape[0] - 1, 128, dtype=int)
                        image_feat = image_feat[sampled_frames]
                    new_encoded_image_features.append(image_feat)
                else:
                    new_encoded_image_features.append(self.get_model().mm_projector(encoded_image_features[idx]))
            
            encoded_image_features = new_encoded_image_features
            
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            
            if mm_patch_merge_type == "flat":
                # get kv image features and position ids
                kv_image_features = [x.flatten(0, 1) for x in encoded_image_features]  # (27x27, 2048)
                kv_position_ids_list = [torch.arange(x.shape[0], device=x.device) for x in kv_image_features]
                # get q image features and position ids
                height = int(math.sqrt(kv_image_features[0].shape[0]))
                q_image_features, q_position_ids_list = self.q_vision_sampling(
                    [x.reshape(height, height, x.shape[-1]) for x in kv_image_features],
                    [x.reshape(height, height) for x in kv_position_ids_list])
                
            elif mm_patch_merge_type.startswith("spatial"):
                q_image_features = []
                kv_image_features = []
                q_position_ids_list = []
                kv_position_ids_list = []
                assert len(modalities) == len(encoded_image_features)
                for image_idx, (modality, image_feature, cur_input_ids, cur_labels) in enumerate(zip(modalities, encoded_image_features, input_ids, labels)):
                    if 'video' in modality.keys():  # video operations
                        if mm_newline_position == "grid":
                            height = width = int(math.sqrt(image_feature.shape[1]))
                            num_images = image_feature.shape[0]
                            # kv: Memory Repository
                            kv_image_feature = image_feature.reshape(num_images, height, width, -1)
                            kv_position_ids = torch.arange(num_images*height*width, device=image_feature.device).reshape(num_images, height, width)
                            
                            # q: Spatial Forgetting
                            q_image_feature, q_position_ids = self.q_vision_sampling(kv_image_feature, kv_position_ids, ratio=0.25)
                            
                            kv_image_feature = kv_image_feature.flatten(1, 2)
                            kv_position_ids = kv_position_ids.flatten(1, 2)
                            
                            if "unpad" in mm_patch_merge_type:
                                q_image_feature = torch.cat([q_image_feature, self.model.image_newline[None, None, :].expand(q_image_feature.shape[0], 1, -1)], dim=1)
                                kv_image_feature = torch.cat([kv_image_feature, self.model.image_newline[None, None, :].expand(kv_image_feature.shape[0], 1, -1)], dim=1)
                            
                            kv_position_ids = torch.arange(num_images*(height*width+1), device=image_feature.device).reshape(num_images, height*width+1)
                            q_position_ids += torch.arange(q_position_ids.shape[0], dtype=q_position_ids.dtype, device=q_position_ids.device).unsqueeze(1)
                            q_position_ids = torch.cat([q_position_ids, kv_position_ids[:, -1].unsqueeze(-1)], dim=-1)

                            # q: Temporal Forgetting
                            total_frames = q_image_feature.shape[0]
                            sampled_frm = min(512, total_frames // 4)
                            sampled_frames = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
                            if total_frames - 1 not in sampled_frames:
                                sampled_frames = np.append(sampled_frames, total_frames - 1)
                            q_image_feature = q_image_feature[sampled_frames]
                            q_position_ids = q_position_ids[sampled_frames]
                            
                            q_image_features.append(q_image_feature.flatten(0, 1))
                            kv_image_features.append(kv_image_feature.flatten(0, 1))
                            q_position_ids_list.append(q_position_ids.flatten(0, 1))
                            kv_position_ids_list.append(kv_position_ids.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif 'multi_image' in modality.keys():  # multi patches and multi images operations
                        height = width = self.get_vision_tower().num_patches_per_side
                        num_images = image_feature.shape[0]
                        kv_image_feature = image_feature.reshape(num_images, height, width, -1)
                        kv_position_ids = torch.arange(num_images*height*width, device=image_feature.device).reshape(num_images, height, width)
                        q_image_feature, q_position_ids = self.q_vision_sampling(kv_image_feature, kv_position_ids)
                        kv_image_feature = kv_image_feature.flatten(1, 2)
                        kv_position_ids = kv_position_ids.flatten(1, 2)
                        
                        if "unpad" in mm_patch_merge_type:
                            q_image_feature = torch.cat([q_image_feature, self.model.image_newline[None, None, :].expand(q_image_feature.shape[0], 1, -1)], dim=1)
                            kv_image_feature = torch.cat([kv_image_feature, self.model.image_newline[None, None, :].expand(kv_image_feature.shape[0], 1, -1)], dim=1)
                        
                        image_newline_token_id = kv_position_ids.max() + 1
                        q_position_ids = torch.cat([q_position_ids, torch.tensor([image_newline_token_id], device=q_position_ids.device).reshape(1, 1).expand(q_position_ids.shape[0], 1)], dim=1)
                        kv_position_ids = torch.cat([kv_position_ids, torch.tensor([image_newline_token_id], device=kv_position_ids.device).reshape(1, 1).expand(kv_position_ids.shape[0], 1)], dim=1)
                        
                        q_image_features.append(q_image_feature)
                        kv_image_features.append(kv_image_feature.flatten(0, 1))
                        q_position_ids_list.append(q_position_ids)
                        kv_position_ids_list.append(kv_position_ids.flatten(0, 1))
                    elif 'image' in modality.keys() and image_feature.shape[0] > 1:  # multi patches and multi images operations
                        base_image_feature, unpad_image_feature = self.multi_image_resize(image_feature, image_sizes[image_idx][0], image_aspect_ratio, mm_patch_merge_type, self.get_vision_tower().num_patches_per_side)
                        base_height, base_width = base_image_feature.shape[:2]
                        unpad_height, unpad_width = unpad_image_feature.shape[:2]
                        unpad_position_ids = torch.arange(unpad_height*unpad_width, device=unpad_image_feature.device).reshape(unpad_height, unpad_width)
                        
                        # nearest neighbor resampling: resize unpad_position_ids -> base_position_ids
                        row_indices = torch.linspace(0, unpad_height - 1, steps=base_height).round().long().to(unpad_image_feature.device)
                        col_indices = torch.linspace(0, unpad_width - 1, steps=base_width).round().long().to(unpad_image_feature.device)
                        base_position_ids = unpad_position_ids[row_indices, :][:, col_indices]
                        # q image features
                        q_base_image_features, q_base_position_ids = self.q_vision_sampling(base_image_feature.unsqueeze(0), base_position_ids.unsqueeze(0))
                        q_unpad_image_features, q_unpad_position_ids = self.q_vision_sampling(unpad_image_feature.unsqueeze(0), unpad_position_ids.unsqueeze(0))
                        
                        # add image newline tokens to kv image features
                        add_image_newline_positions = unpad_position_ids[1:, 0]
                        image_newline_token_id = unpad_position_ids.max() + 1
                        unpad_position_ids = torch.cat((unpad_position_ids, torch.tensor([image_newline_token_id], device=unpad_image_feature.device).reshape(1, 1).expand(unpad_image_feature.shape[0], 1)), dim=1)
                        unpad_image_feature = torch.cat((unpad_image_feature, self.model.image_newline[None, None, :].expand(unpad_image_feature.shape[0], 1, unpad_image_feature.shape[-1])), dim=1)
                        # kv image features
                        kv_image_feature = torch.cat([base_image_feature.flatten(0, 1), unpad_image_feature.flatten(0, 1)], dim=0)
                        kv_position_ids = torch.cat([base_position_ids.reshape(-1), unpad_position_ids.reshape(-1)], dim=0)
                        
                        # add image newline tokens to q image features
                        q_unpad_image_features, q_unpad_position_ids = self.add_grid_image_newline(q_unpad_image_features[0], q_unpad_position_ids[0], add_image_newline_positions, image_newline_token_id)
                        q_image_feature = torch.cat([q_base_image_features[0], q_unpad_image_features], dim=0)
                        q_position_ids = torch.cat([q_base_position_ids[0], q_unpad_position_ids], dim=0)
                        
                        q_image_features.append(q_image_feature)
                        kv_image_features.append(kv_image_feature)
                        q_position_ids_list.append(q_position_ids)
                        kv_position_ids_list.append(kv_position_ids)
                    else:  # single image operations
                        kv_image_feature = image_feature.flatten(0, 1)  # (27x27, 2048)
                        kv_position_ids = torch.arange(kv_image_feature.shape[0], device=kv_image_feature.device)
                        # get q image features and position ids
                        height = int(math.sqrt(kv_image_feature.shape[0]))
                        q_image_feature, q_position_ids = self.q_vision_sampling(kv_image_feature.reshape(height, height, -1).unsqueeze(0), kv_position_ids.reshape(height, height).unsqueeze(0))
                        kv_image_feature = self.single_image_resize([kv_image_feature], mm_patch_merge_type)
                        q_image_feature = self.single_image_resize(q_image_feature, mm_patch_merge_type)
                        # add unpad position id
                        unpad_position_id = kv_position_ids.max() + 1
                        kv_position_ids = torch.cat([kv_position_ids, torch.tensor([unpad_position_id], device=kv_position_ids.device)])
                        q_position_ids = torch.cat([q_position_ids[0], torch.tensor([unpad_position_id], device=kv_position_ids.device)])
                        
                        q_image_features.append(q_image_feature)
                        kv_image_features.append(kv_image_feature)
                        q_position_ids_list.append(q_position_ids)
                        kv_position_ids_list.append(kv_position_ids)
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            raise NotImplementedError
            
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
            
        ############################# MemAug #############################
        assert len(q_image_features) == len(kv_image_features)
        assert len(kv_image_features) == len(modalities)
        assert len(modalities) == len(input_ids)
        assert len(input_ids) == len(labels)
        image_features = []
        for image_idx, (q_image_feature, kv_image_feature, q_position_ids, kv_position_ids, modality, cur_input_ids, cur_labels) in enumerate(zip(q_image_features, kv_image_features, q_position_ids_list, kv_position_ids_list, modalities, input_ids, labels)):
            if 'video' in modality.keys() or 'image' in modality.keys() or 'text' in modality.keys():  # num_images == 1
                extended_q_image_feature = []
                extended_q_position_ids = []
                max_position_id = -1
                if 'text' in modality.keys():
                    num_images = 1
                    # add IMAGE_TOKEN_INDEX at the end
                    _cur_input_ids = torch.cat([cur_input_ids, torch.tensor([IMAGE_TOKEN_INDEX], device=cur_input_ids.device)])
                else:
                    num_images = modality['video'] if 'video' in modality.keys() else modality['image']
                    _cur_input_ids = cur_input_ids
                image_token_indices = [-1] + torch.where(_cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [_cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(_cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
                
                image_position_ids = []
                for i in range(num_images + 1):
                    extended_q_image_feature.append(cur_input_embeds_no_im[i][cur_labels_noim[i] == IGNORE_INDEX])  # exclude the answer tokens
                    new_q_position_ids = torch.arange(
                        cur_input_embeds_no_im[i][cur_labels_noim[i] == IGNORE_INDEX].shape[0],
                        device=q_image_feature.device,
                        dtype=q_position_ids.dtype) + max_position_id + 1
                    if len(extended_q_position_ids) == 0:
                        extended_q_position_ids = new_q_position_ids
                    else:
                        extended_q_position_ids = torch.cat([extended_q_position_ids, new_q_position_ids])
                    max_position_id = extended_q_position_ids.max()
                    if i < num_images:
                        extended_q_image_feature.append(q_image_feature)
                        q_position_ids = q_position_ids + max_position_id + 1
                        image_position_ids.append(torch.arange(q_position_ids.shape[0], device=q_image_feature.device, dtype=q_position_ids.dtype) + extended_q_position_ids.shape[0])
                        extended_q_position_ids = torch.cat([extended_q_position_ids, q_position_ids])
                        max_position_id = extended_q_position_ids.max()
                extended_q_image_feature = torch.cat([x.to(self.device) for x in extended_q_image_feature])
                image_position_ids = torch.cat(image_position_ids)
                
                # MemAug module
                image_feat = self.get_model().vision_resampler(extended_q_image_feature, kv_image_feature, q_position_ids=extended_q_position_ids, k_position_ids=kv_position_ids)
                assert image_position_ids.max() < image_feat.shape[0]
                # exclude language token
                image_feat = image_feat[image_position_ids, :]
                assert image_feat.shape[0] == q_image_feature.shape[0]
                image_features.append(image_feat)
            elif 'multi_image' in modality.keys():  # num_images > 1
                extended_q_image_feature = []
                extended_q_position_ids = []
                max_position_id = -1
                num_images = modality['multi_image']
                
                # get input_ids and labels with no image tokens
                _cur_input_ids = cur_input_ids
                image_token_indices = [-1] + torch.where(_cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [_cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(_cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
                
                # add image tokens into input_ids
                image_position_ids = []
                for i in range(num_images + 1):
                    extended_q_image_feature.append(cur_input_embeds_no_im[i][cur_labels_noim[i] == IGNORE_INDEX])  # exclude the answer tokens
                    new_q_position_ids = torch.arange(
                        cur_input_embeds_no_im[i][cur_labels_noim[i] == IGNORE_INDEX].shape[0],
                        device=q_image_feature.device,
                        dtype=q_position_ids.dtype) + max_position_id + 1
                    if len(extended_q_position_ids) == 0:
                        extended_q_position_ids = new_q_position_ids
                    else:
                        extended_q_position_ids = torch.cat([extended_q_position_ids, new_q_position_ids])
                    max_position_id = extended_q_position_ids.max()
                    if i < num_images:
                        extended_q_image_feature.append(q_image_feature[i])
                        _q_position_ids = q_position_ids[i] + max_position_id + 1
                        image_position_ids.append(torch.arange(_q_position_ids.shape[0], device=q_image_feature.device, dtype=q_position_ids.dtype) + extended_q_position_ids.shape[0])
                        extended_q_position_ids = torch.cat([extended_q_position_ids, _q_position_ids])
                        max_position_id = extended_q_position_ids.max()
                extended_q_image_feature = torch.cat([x.to(self.device) for x in extended_q_image_feature])
                image_position_ids = torch.cat(image_position_ids)
                
                # compressor
                image_feat = self.get_model().vision_resampler(extended_q_image_feature, kv_image_feature, q_position_ids=extended_q_position_ids, k_position_ids=kv_position_ids)
                assert image_position_ids.max() < image_feat.shape[0]
                # exclude language token
                image_feat = image_feat[image_position_ids, :].reshape(q_image_feature.shape[0], -1, image_feat.shape[-1])
                image_features.extend([feat for feat in image_feat])
            else:
                raise NotImplementedError
        ######################################################################

        ######################### Add query tokens ###########################
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        ######################################################################

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")
        
        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        max_len = max([x.shape[0] for x in new_input_embeds])
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

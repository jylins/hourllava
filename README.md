# <img src="assets/icon.png" style="vertical-align: -10px;" :height="50px" width="50px"> Unleashing Hour-Scale Video Training for Long Video-Language Understanding

This repository is the official PyTorch implementation of Hour-LLaVA.

[[Project Website](https://videomarathon.github.io/)] [[Paper](https://arxiv.org/abs/xxxx.xxxxx)] [[Dataset](https://huggingface.co/datasets/jylins/videomarathon)] [Model Zoo]



## 1 VideoMarathon Dataset
**VideoMarathon** is a large-scale long video instruction-following dataset with a total duration of approximately **9,700 hours**, comprising **3.3 million QA pairs** across **22 task categories**.
## 1.1 Task Taxonomy
The dataset contains 22 diverse tasks over six fundamental topics, including temporality, spatiality, object, action, scene, and event. These diverse tasks require both *short-form* (yellow tag) and *long-form* (red tag) video comprehension.
![alt text](assets/task_taxonomy.png)
## 1.2 Data Statistics
![alt text](assets/statistics.png)
- **Data Source**: The dataset spans diverse video source domains.
- **Question Type**: The dataset features a wide range of question types for long-form video-language modeling.
- **Video Duration**: The dataset consists of long videos ranging from three minutes to one hour.
- **Event Counting**: The dataset includes complex video content reflected by the number of events per video.

## 1.3 Data Comparison
The comparison between our VideoMarathon and other existing video instruction-following datasets shows that VideoMarathon features a significantly *longer average video length*, *broader duration range*, and *a larger number of QA pairs*.
![alt text](assets/comparison.png)


## 2 Hour-LLaVA: An Efficient Hour-scale Video-LMM
Powered by memory augmentation, we propose Hour-LLaVA, an efficient video-language model capable of modeling hour-long videos at 1 FPS. It comprises three key modules: a video encoder, a memory augmentation module (i.e., MemAug), and an LLM decoder.
![alt text](assets/hour_llava.png)


## 3 Training Code and Model Zoo
Coming soon..

## 4 Citation
```bash
@article{lin2025unleashing,
  author    = {Lin, Jingyang and Wu, Jialian and Sun, Ximeng and Wang, Ze and Liu, Jiang and Chen, Hao and Luo, Jiebo and Liu, Zicheng and Barsoum, Emad},
  title     = {Unleashing Hour-Scale Video Training for Long Video-Language Understanding},
  journal   = {arXiv preprint arXiv:2506.05332},
  year      = {2025},
}
```

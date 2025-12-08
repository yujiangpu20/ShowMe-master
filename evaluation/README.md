
# Evaluation Suite

This is a step-by-step guidance to conduct evaluation for Image/Video generaton on different dataset.

## 1. Image Evaluation

To evaluate generated images:

```
python eval_img.py \
  --dataset ssv2 \
  --meta_file /path/to/meta_file \
  --video_dir /path/to/videos \
  --save_dir /path/to/save_dir
```

This computes all six metrics reported in our paper (i.e., CLIP-I, CLIP-T, DINO-I, FID, PSNR, and LPIPS). At present, it supports ```ssv2``` and ```epic100``` datasets. For ```ego4d``` we follow a different evaluation protocol by [LEGO](https://github.com/BolinLai/LEGO). Please refer to it for more details.


## 2. Video Evaluation

### 2.1 FVD/FID/CLIP Score

To evaluate generated videos:

```
python eval_vid.py \
  --dataset ssv2 \
  --meta_file /path/to/meta_file \
  --video_dir /path/to/generated_videos \
  --real_video_dir /path/to/real_videos \
  --save_dir /path/to/save_dir
```

### 2.1 ViCLIP for SSv2

To compute the ViCLIP score of ```ssv2``` dataset:

```
python eval_viclip.py \
  --video_dir /path/to/videos \
  --meta_file /path/to/metadata.jsonl \
  --save_dir /path/to/save/results \
  --batch_size 32 \
  --n_frames 8 \
  --num_workers 8 \
  --gpu_id 0
```

### 2.2 EgoVLP for Epic100

To compute the EgoVLP score of ```epic100``` dataset:
```
# Example for EPIC-Kitchens-100
python eval_egovlp.py \
  --dataset epic100 \
  --gen_path /path/to/generated/videos \
  --meta_path /path/to/epic_meta.csv \
  --batch_size 32
```

### 2.3 VQA Score

To compute the VQA Score, first create a new conda environment following the instructions in the [VQAScore](https://github.com/linzhiqiu/t2v_metrics) repository and download the pretrained MLLM checkpoints.

Subsequently, activate the environment and run the following command:
```
python vqa_score.py \
  --dataset ssv2 \
  --video_dir /path/to/generated/videos \
  --meta_file /path/to/metadata.jsonl \
  --save_dir /path/to/save/vqa_results \
  --batch_size 16 \
  --gpu_id 0
```

Modify the dataset names (i.e., ```ssv2```, ```epic100```, ```ego4d```) along with the corresponding data and metadata file paths.

### 2.4 Motion Score

We use the standard [VBench](https://github.com/Vchitect/VBench) evaluation suite to compute **motion smoothness** and **dynamic degree**. First, creat a new conda environment according to the official instructions.

Subsequently, activate the environment and run the following command:

```
bash vbench_eval.sh
```

Please modify the ```VIDEO_DIR``` and ```OUTPUT_DIR``` inside the shell script to your local path.



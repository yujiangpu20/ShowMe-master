

## Image Evaluation

To evaluate generated images:

```
python eval_img.py \
  --dataset ssv2 \
  --meta_file /path/to/meta_file \
  --video_dir /path/to/videos \
  --save_dir /path/to/save_dir
```

## Video Evaluation

To evaluate generated videos:

```
python eval_vid.py \
  --dataset ssv2 \
  --meta_file /path/to/meta_file \
  --video_dir /path/to/generated_videos \
  --real_video_dir /path/to/real_videos \
  --save_dir /path/to/save_dir
```


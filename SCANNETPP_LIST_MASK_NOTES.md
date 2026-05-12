# 本版针对 ScanNet++ list 与不规则 mask 的修改

## 1. list 路径解析

支持如下路径：

```text
/root/.../S_imgs/0a5c013435_part1/dslr/resized_undistorted_images/DSC01752.JPG
/root/.../S_imgs/0a5c013435_part1/dslr/resized_undistorted_images/DSC01753.JPG
```

现在 `core/dataset.py` 会将 `dslr` 前一级目录作为序列名：

```text
video_name = 0a5c013435_part1
file_name  = DSC01752.JPG
```

pkl 使用同 stem 匹配，例如：

```text
DSC01752.JPG <-> DSC01752.pkl
```

## 2. mask_list 自然顺序匹配

不再要求 mask 与图片同名。`mask_list` 会按自然顺序排序，然后按 frame index 取 mask：

```python
mask_path = mask_paths[frame_idx % len(mask_paths)]
```

默认配置：

```json
"mask_match_mode": "frame_index"
```

如果希望每个 sampled clip 内从第 0 张 mask 开始顺序配，可以改为：

```json
"mask_match_mode": "clip_order"
```

## 3. debug

先运行：

```bash
python debug_dataset.py
```

如果能看到 frame/mask/line tensor shape，再开始训练：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/train_propainter.json
```

## 4. ProPainter 预训练权重

默认配置写了：

```json
"gen_path": "weights/ProPainter.pth"
```

如果你的权重路径不同，请修改为实际路径。若路径不存在，程序会给出 warning，并从随机初始化开始训练。

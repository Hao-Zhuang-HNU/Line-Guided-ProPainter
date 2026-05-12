# Line-Guided ProPainter 修改说明

本版本在 ProPainter 的图像补全 generator 中加入了 pkl 线框图引导，用于快速验证“结构线框是否能提升视频补全的结构恢复能力”。

## 主要改动

### 1. `model/propainter.py`

- 新增 `LineEncoder`：输入 `[B*T, 1, H, W]` 的二值线框图，输出与 ProPainter Encoder 对齐的 `[B*T, 128, H/4, W/4]` 特征。
- 在 `InpaintGenerator.forward()` 中新增可选参数：

```python
line_guidance=None
```

- 如果传入 `line_guidance`，执行：

```text
RGB/mask Encoder feature + LineEncoder feature -> line_fuse -> residual fusion
```

- 如果 `line_guidance=None`，则跳过线框分支，保持原始 ProPainter 行为。
- 加载官方 ProPainter checkpoint 时使用 `strict=False`，因此新增的 `line_encoder` 与 `line_fuse` 会随机初始化，原 ProPainter 权重仍可加载。

### 2. `core/dataset.py`

- 新增 pkl 线框读取与渲染函数：
  - `load_lines_from_pkl()`
  - `render_lines_to_pil()`
- 兼容两类 pkl 格式：
  - `{"lines": [[x1, y1, x2, y2], ...], "scores": ...}`
  - 直接保存的 `[[x1, y1, x2, y2], ...]`
- 兼容 numpy 2.x 生成的 pickle 在 numpy 1.x 环境反序列化时的 `numpy._core` 问题。
- 默认按照你上传的 `visualize_pkl.py` 逻辑处理：
  - 默认交换 x/y；
  - 默认采用图像坐标系 y 轴方向；
  - 黑底白线，输出 `[0, 1]` tensor。
- `TrainDataset` 现在返回：

```python
frame_tensors, mask_tensors, line_tensors, flows_f, flows_b, video_name
```

### 3. `core/trainer.py`

- 训练时将 `line_tensors` 传入 generator：

```python
pred_imgs = self.netG(..., line_guidance=line_tensors)
```

- 新增 `line-weighted reconstruction loss`。
- 新增 `Sobel structure loss`。
- 如果 `line_root` 不存在，则自动关闭 line guidance 和新增结构损失。
- 修复单卡非 DDP 情况下调用 `self.netG.module.img_propagation` 可能报错的问题。

### 4. `inference_propainter.py`

新增推理参数：

```bash
--line /path/to/pkl_lines
--line_width 1
--line_no_swap_xy
--line_no_invert_y
```

示例：

```bash
python inference_propainter.py \
  --video inputs/object_removal/MV_test/MV_imgs/000024c8/images \
  --mask inputs/object_removal/MV_test/MV_masks/dilate10/masks/000024c8/images \
  --line /path/to/pkl_lines/000024c8 \
  --save_frames
```

注意：如果直接使用官方 ProPainter 权重，新增 line branch 是随机初始化的。正式使用 `--line` 推理前，应先用 line-guided 版本 fine-tune 得到新的 generator 权重。

## 训练配置

`configs/train_propainter.json` 已增加：

```json
"line_root": "your_pkl_line_root",
"line_width": 1,
"line_swap_xy": true,
"line_invert_y": true
```

以及损失权重：

```json
"line_weighted_weight": 1.0,
"line_loss_alpha": 3.0,
"sobel_weight": 0.05,
"sobel_line_alpha": 2.0
```

## 推荐第一轮实验

1. 使用官方 ProPainter 权重作为初始化。
2. 使用 GT pkl line 做 upper bound：验证结构线框是否能被 ProPainter 利用。
3. 再换成 predicted pkl line：验证真实推理设置下是否有效。
4. 第一轮建议冻结 flow completion，仅 fine-tune inpainting generator。

## 线框目录格式

建议：

```text
your_pkl_line_root/
  video_001/
    00000.pkl
    00001.pkl
    ...
```

文件名 stem 需要和图像帧一致，例如：

```text
frames/video_001/00000.png
lines/video_001/00000.pkl
```

# 本次修复说明

针对 debug_dataset.py 输出：

```text
dataset length: 1
first 10 video_names: ['resized_undistorted_images']
use_mask_list: None num_masks: 0
AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'
```

修复了三个问题：

1. **ScanNet++ list 路径解析**
   - 支持 `/.../S_imgs/0a5c013435_part1/dslr/resized_undistorted_images/DSC01752.JPG`。
   - 现在会识别 `video_name=0a5c013435_part1`，不会再识别成 `resized_undistorted_images`。
   - 同时支持 `f1e01af60a`、`0a5c013435_part1` 等 scene id。

2. **mask_list 读取更鲁棒**
   - `mask_list` 可以是 txt 文件，也可以是目录。
   - 如果是 txt 文件，则每行一个 mask 路径。
   - 如果是目录，则递归收集 png/jpg/jpeg。
   - 按自然顺序排序，然后用 frame index 或 clip order 匹配。

3. **Matplotlib 兼容性**
   - `FigureCanvasAgg.tostring_rgb()` 在部分新版 Matplotlib 中不存在。
   - 已在 `core/utils.py` 中加入 `buffer_rgba()` fallback。
   - 即使 mask_list 没有读到，回退到随机 mask 时也不会因为该 API 报错。

建议先运行：

```bash
python debug_dataset.py
```

期望结果：

```text
dataset length: 大于 1
first 10 video_names: ['0a5c013435_part1', ...]
use_line: True
use_mask_list: True num_masks: 大于 0
```

若 `use_mask_list` 仍为 False，请检查 `configs/train_propainter.json` 中的 `mask_list` 路径是否存在。可以把 `mask_list` 写成绝对路径。

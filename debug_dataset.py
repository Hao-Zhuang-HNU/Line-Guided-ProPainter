import json
import time
from core.dataset import TrainDataset

cfg = json.load(open('configs/train_propainter.json', 'r'))
dataset = TrainDataset(cfg['train_data_loader'])

print('dataset length:', len(dataset))
print('first 10 video_names:', dataset.video_names[:10])
print('use_line:', getattr(dataset, 'use_line', None))
print('use_mask_list:', getattr(dataset, 'use_mask_list', None), 'num_masks:', len(getattr(dataset, 'mask_paths', [])))

for i in range(min(3, len(dataset))):
    print(f'\n[DEBUG] loading sample {i}', flush=True)
    t0 = time.time()
    data = dataset[i]
    print(f'[DEBUG] loaded sample {i}, cost={time.time() - t0:.3f}s', flush=True)
    for j, x in enumerate(data):
        if hasattr(x, 'shape'):
            print(j, x.shape)
        else:
            print(j, type(x), x)

import os
import sys
import json
import random
import pickle

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from utils.file_client import FileClient
from utils.img_util import imfrombytes
from utils.flow_util import resize_flow, flowread
from core.utils import (create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip,GroupRandomHorizontalFlowFlip)


def _shim_numpy_core_for_pickle():
    """Compatibility shim for pkl files generated under numpy 2.x."""
    try:
        import numpy as np
        import numpy.core as ncore
        sys.modules.setdefault("numpy._core", ncore)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    except Exception:
        pass


def load_lines_from_pkl(pkl_path):
    """Load wireframe line segments from pkl.

    Supported formats:
    - {"lines": [[x1,y1,x2,y2], ...], "scores": ...}
    - directly [[x1,y1,x2,y2], ...]
    """
    _shim_numpy_core_for_pickle()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "lines" in data:
        lines = data["lines"]
    else:
        lines = data

    if isinstance(lines, np.ndarray):
        lines = lines.tolist()

    cleaned = []
    if isinstance(lines, (list, tuple)):
        for ln in lines:
            if isinstance(ln, np.ndarray):
                ln = ln.tolist()
            if isinstance(ln, (list, tuple)) and len(ln) == 4:
                x1, y1, x2, y2 = ln
                cleaned.append((float(x1), float(y1), float(x2), float(y2)))
    return cleaned


def render_lines_to_pil(lines, size, line_width=1, swap_xy=True, invert_y=True):
    """Render pkl wireframe to a black-background, white-line PIL L image.

    This mirrors the uploaded visualize_pkl.py convention: by default x/y are
    swapped when plotting, and the y-axis is treated like image coordinates.
    """
    W, H = int(size[0]), int(size[1])
    canvas = np.zeros((H, W), dtype=np.uint8)
    if not lines:
        return Image.fromarray(canvas, mode='L')

    max_v = 0.0
    for x1, y1, x2, y2 in lines:
        max_v = max(max_v, abs(x1), abs(y1), abs(x2), abs(y2))
    normalized = max_v <= 1.5

    thickness = max(1, int(round(line_width)))
    for x1, y1, x2, y2 in lines:
        if normalized:
            x1, x2 = x1 * (W - 1), x2 * (W - 1)
            y1, y2 = y1 * (H - 1), y2 * (H - 1)

        if swap_xy:
            px1, py1 = y1, x1
            px2, py2 = y2, x2
        else:
            px1, py1 = x1, y1
            px2, py2 = x2, y2

        if not invert_y:
            py1 = (H - 1) - py1
            py2 = (H - 1) - py2

        p1 = (int(round(np.clip(px1, 0, W - 1))), int(round(np.clip(py1, 0, H - 1))))
        p2 = (int(round(np.clip(px2, 0, W - 1))), int(round(np.clip(py2, 0, H - 1))))
        cv2.line(canvas, p1, p2, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas, mode='L')


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']
        self.flow_root = args['flow_root']
        self.line_root = args.get('line_root', None)
        self.use_line = self.line_root is not None and os.path.exists(self.line_root)
        self.line_width = args.get('line_width', 1)
        self.line_swap_xy = args.get('line_swap_xy', True)
        self.line_invert_y = args.get('line_invert_y', True)
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)
        
        json_path = os.path.join('./datasets', args['name'], 'train.json')

        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict.keys()))

        # self.video_names = sorted(os.listdir(self.video_root))
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list
                

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        masks = []
        lines = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)
            masks.append(all_masks[idx])

            if self.use_line:
                line_stem = os.path.splitext(frame_list[idx])[0] + '.pkl'
                line_path = os.path.join(self.line_root, video_name, line_stem)
                if os.path.exists(line_path):
                    line_segments = load_lines_from_pkl(line_path)
                else:
                    line_segments = []
                line_img = render_lines_to_pil(
                    line_segments,
                    self.size,
                    line_width=self.line_width,
                    swap_xy=self.line_swap_xy,
                    invert_y=self.line_invert_y)
            else:
                line_img = Image.fromarray(np.zeros((self.h, self.w), dtype=np.uint8), mode='L')
            lines.append(line_img)

            if len(frames) <= self.num_local_frames-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_local_frames: # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    masks.reverse()
                    lines.reverse()
                    if self.load_flow:
                        flows_f.reverse()
                        flows_b.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_
                
        if self.load_flow:
            v = random.random()
            if v < 0.5:
                frames = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in frames]
                masks = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in masks]
                lines = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in lines]
                flows_f = [ff[:, ::-1] * [-1.0, 1.0] for ff in flows_f]
                flows_b = [fb[:, ::-1] * [-1.0, 1.0] for fb in flows_b]
        else:
            v = random.random()
            if v < 0.5:
                frames = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in frames]
                masks = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in masks]
                lines = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in lines]

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        line_tensors = self._to_tensors(lines)
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        # img [-1,1] mask [0,1]
        if self.load_flow:
            return frame_tensors, mask_tensors, line_tensors, flows_f, flows_b, video_name
        else:
            return frame_tensors, mask_tensors, line_tensors, 'None', 'None', video_name


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args['size']

        self.video_root = args['video_root']
        self.mask_root = args['mask_root']
        self.flow_root = args['flow_root']

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)
        self.video_names = sorted(os.listdir(self.mask_root))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            self.video_dict[v] = v_len
            self.frame_dict[v] = frame_list

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        selected_index = list(range(self.video_dict[video_name]))

        # read video frames
        frames = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            frame_path = os.path.join(self.video_root, video_name, frame_list[idx])

            img_bytes = self.file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)

            mask_path = os.path.join(self.mask_root, video_name, str(idx).zfill(5) + '.png')
            mask = Image.open(mask_path).resize(self.size, Image.NEAREST).convert('L')

            # origin: 0 indicates missing. now: 1 indicates missing
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)

            m = cv2.dilate(m,
                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                           iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

            if len(frames) <= len(selected_index)-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

        # normalizate, to tensors
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        if self.load_flow:
            return frame_tensors, mask_tensors, flows_f, flows_b, video_name, frames_PIL
        else:
            return frame_tensors, mask_tensors, 'None', 'None', video_name
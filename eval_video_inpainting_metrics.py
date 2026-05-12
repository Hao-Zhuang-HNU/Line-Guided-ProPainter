#!/usr/bin/env python3
# eval_video_inpainting_metrics.py
# Metrics: PSNR-all/hole, SSIM-all/hole, LPIPS, FID, VFID(proxy), Ewarp-all/hole, temporal LPIPS

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import linalg

try:
    import cv2
except Exception:
    cv2 = None

try:
    import lpips  # pip install lpips
except Exception:
    lpips = None

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def natural_key(p: Path) -> List:
    import re
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=natural_key)
    return files


def pil_to_np_rgb01(pil_img: Image.Image, size: Optional[int] = None) -> np.ndarray:
    img = pil_img.convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr  # HWC, [0,1]


def load_mask(mask_path: Path, size: Optional[int] = None) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    if size is not None and m.size != (size, size):
        m = m.resize((size, size), resample=Image.NEAREST)
    arr = np.asarray(m).astype(np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)  # HW, 1=hole


def psnr_hole(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask[..., None]
    denom = float(np.sum(m)) * 3.0
    if denom < 1e-6:
        return float("nan")
    mse = float(np.sum(((pred - gt) ** 2) * m) / denom)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_hole_bbox(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, pad: int = 8) -> float:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return float("nan")
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad); y1 = min(pred.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(pred.shape[1] - 1, x1 + pad)
    p = pred[y0:y1+1, x0:x1+1, :]
    g = gt[y0:y1+1, x0:x1+1, :]
    return float(structural_similarity(p, g, channel_axis=2, data_range=1.0))


# ----------------------- LPIPS -----------------------
class LPIPSMetric:
    def __init__(self, device: torch.device, net: str = "alex"):
        if lpips is None:
            raise RuntimeError("lpips not installed. pip install lpips")
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred_rgb01: np.ndarray, gt_rgb01: np.ndarray) -> float:
        p = torch.from_numpy(pred_rgb01).permute(2, 0, 1).unsqueeze(0).to(self.device)
        g = torch.from_numpy(gt_rgb01).permute(2, 0, 1).unsqueeze(0).to(self.device)
        p = p * 2.0 - 1.0
        g = g * 2.0 - 1.0
        return float(self.model(p, g).item())


# ----------------------- FID -----------------------
class InceptionPool3(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = torchvision.models.Inception_V3_Weights.DEFAULT
        # some torchvision versions require aux_logits=True when loading pretrained weights
        self.net = torchvision.models.inception_v3(weights=weights, aux_logits=True).to(device).eval()
        self.net.fc = nn.Identity()
        self.preprocess = weights.transforms()  # expects PIL

    @torch.no_grad()
    def forward(self, pil_imgs: List[Image.Image], device: torch.device) -> torch.Tensor:
        xs = [self.preprocess(img) for img in pil_imgs]
        x = torch.stack(xs, dim=0).to(device)
        return self.net(x)  # Nx2048


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


@torch.no_grad()
def fid_features(
    pred_paths: List[Path],
    gt_paths: List[Path],
    pix: Optional[int],
    model: InceptionPool3,
    device: torch.device,
    batch_size: int,
) -> float:
    pred_feats, gt_feats = [], []
    for i in tqdm(range(0, len(pred_paths), batch_size), desc="FID feat"):
        bp = pred_paths[i:i+batch_size]
        bg = gt_paths[i:i+batch_size]
        pred_pils, gt_pils = [], []
        for p, g in zip(bp, bg):
            pi = Image.open(p).convert("RGB")
            gi = Image.open(g).convert("RGB")
            if pix is not None:
                if pi.size != (pix, pix): pi = pi.resize((pix, pix), Image.BILINEAR)
                if gi.size != (pix, pix): gi = gi.resize((pix, pix), Image.BILINEAR)
            pred_pils.append(pi)
            gt_pils.append(gi)
        pred_feats.append(model(pred_pils, device).cpu().numpy())
        gt_feats.append(model(gt_pils, device).cpu().numpy())

    P = np.concatenate(pred_feats, axis=0)
    G = np.concatenate(gt_feats, axis=0)
    mu_p, sig_p = np.mean(P, axis=0), np.cov(P, rowvar=False)
    mu_g, sig_g = np.mean(G, axis=0), np.cov(G, rowvar=False)
    return compute_frechet_distance(mu_p, sig_p, mu_g, sig_g)


# ----------------------- VFID (proxy) -----------------------
class VideoBackbone(nn.Module):
    def __init__(self, name: str, device: torch.device):
        super().__init__()
        if name == "r3d_18":
            weights = torchvision.models.video.R3D_18_Weights.DEFAULT
            self.net = torchvision.models.video.r3d_18(weights=weights)
        elif name == "mc3_18":
            weights = torchvision.models.video.MC3_18_Weights.DEFAULT
            self.net = torchvision.models.video.mc3_18(weights=weights)
        elif name == "r2plus1d_18":
            weights = torchvision.models.video.R2Plus1D_18_Weights.DEFAULT
            self.net = torchvision.models.video.r2plus1d_18(weights=weights)
        else:
            raise ValueError(f"Unknown VFID backbone: {name}")

        self.net.fc = nn.Identity()
        self.net.eval().to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,3,T,112,112
        return self.net(x)


# torchvision video models commonly use Kinetics-400 normalization
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def make_clips(frame_paths: List[Path], clip_len: int, stride: int) -> List[List[Path]]:
    clips = []
    if len(frame_paths) < clip_len:
        return clips
    for s in range(0, len(frame_paths) - clip_len + 1, stride):
        clips.append(frame_paths[s:s+clip_len])
    return clips


def read_clip_uint8_T_H_W_C(paths: List[Path], pix: Optional[int]) -> np.ndarray:
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if pix is not None and img.size != (pix, pix):
            img = img.resize((pix, pix), Image.BILINEAR)
        frames.append(np.asarray(img).astype(np.uint8))
    return np.stack(frames, axis=0)  # T,H,W,C


def preprocess_video_batch_uint8(bthwc: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Input : uint8 tensor, shape B,T,H,W,C
    Output: float tensor, shape B,3,T,112,112 normalized (kinetics)
    """
    x = bthwc.to(device).float() / 255.0  # B,T,H,W,C
    x = x.permute(0, 4, 1, 2, 3).contiguous()  # B,C,T,H,W

    B, C, T, H, W = x.shape
    x2 = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)  # (B*T),C,H,W
    x2 = F.interpolate(x2, size=(112, 112), mode="bilinear", align_corners=False)
    x = x2.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4).contiguous()  # B,C,T,112,112

    x = (x - KINETICS_MEAN.to(device)) / KINETICS_STD.to(device)
    return x


@torch.no_grad()
def compute_vfid_proxy(
    pred_frames: List[Path],
    gt_frames: List[Path],
    vb: VideoBackbone,
    device: torch.device,
    pix: Optional[int],
    clip_len: int,
    stride: int,
    batch_size: int = 8,
) -> float:
    clips_idx = make_clips(list(range(len(pred_frames))), clip_len, stride)
    if len(clips_idx) < 2:
        return float("nan")

    pred_feats, gt_feats = [], []
    for i in tqdm(range(0, len(clips_idx), batch_size), desc="VFID clips"):
        batch = clips_idx[i:i+batch_size]
        pred_batch, gt_batch = [], []
        for inds in batch:
            pclip = read_clip_uint8_T_H_W_C([pred_frames[j] for j in inds], pix)
            gclip = read_clip_uint8_T_H_W_C([gt_frames[j] for j in inds], pix)
            pred_batch.append(pclip)
            gt_batch.append(gclip)

        pred_np = np.stack(pred_batch, axis=0)  # B,T,H,W,C
        gt_np = np.stack(gt_batch, axis=0)

        pred_x = preprocess_video_batch_uint8(torch.from_numpy(pred_np), device)
        gt_x = preprocess_video_batch_uint8(torch.from_numpy(gt_np), device)

        pred_feats.append(vb(pred_x).cpu().numpy())
        gt_feats.append(vb(gt_x).cpu().numpy())

    P = np.concatenate(pred_feats, axis=0)
    G = np.concatenate(gt_feats, axis=0)
    mu_p, sig_p = np.mean(P, axis=0), np.cov(P, rowvar=False)
    mu_g, sig_g = np.mean(G, axis=0), np.cov(G, rowvar=False)
    return compute_frechet_distance(mu_p, sig_p, mu_g, sig_g)


# ----------------------- Optical flow + Ewarp -----------------------
def farneback_flow(im1_rgb01: np.ndarray, im2_rgb01: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python not found. pip install opencv-python")
    g1 = cv2.cvtColor((im1_rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor((im2_rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow.astype(np.float32)


def warp_with_flow(img_rgb01: np.ndarray, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("opencv-python not found. pip install opencv-python")
    h, w = img_rgb01.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xs + flow[..., 0]).astype(np.float32)
    map_y = (ys + flow[..., 1]).astype(np.float32)
    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    warped = cv2.remap(
        (img_rgb01 * 255).astype(np.uint8),
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype(np.float32) / 255.0
    return warped, valid.astype(np.float32)


def compute_ewarp_and_tlpips(
    pred_frames: List[Path],
    gt_frames: List[Path],
    mask_frames: List[Optional[Path]],
    pix: Optional[int],
    lpips_metric: Optional[LPIPSMetric],
    flow_source: str = "gt",
) -> Dict[str, float]:
    e_all, e_hole = [], []
    tlp_all, tlp_hole = [], []

    N = min(len(pred_frames), len(gt_frames))
    if N < 2:
        return {"Ewarp_all": float("nan"), "Ewarp_hole": float("nan"),
                "tLPIPS_all": float("nan"), "tLPIPS_hole": float("nan")}

    for t in range(N - 1):
        p_t, p_tp1 = pred_frames[t], pred_frames[t+1]
        g_t, g_tp1 = gt_frames[t], gt_frames[t+1]

        pred_t = pil_to_np_rgb01(Image.open(p_t), size=pix)
        pred_tp1 = pil_to_np_rgb01(Image.open(p_tp1), size=pix)
        gt_t = pil_to_np_rgb01(Image.open(g_t), size=pix)
        gt_tp1 = pil_to_np_rgb01(Image.open(g_tp1), size=pix)

        im1, im2 = (gt_t, gt_tp1) if flow_source == "gt" else (pred_t, pred_tp1)
        flow = farneback_flow(im1, im2)

        warped_pred_tp1, valid = warp_with_flow(pred_tp1, flow)
        diff2 = np.mean((pred_t - warped_pred_tp1) ** 2, axis=2)

        denom_all = np.sum(valid)
        if denom_all > 1:
            e_all.append(float(np.sum(diff2 * valid) / denom_all))

        mpath = mask_frames[t]
        if mpath is not None and Path(mpath).exists():
            m = load_mask(Path(mpath), size=pix)
            denom_h = np.sum(valid * m)
            if denom_h > 1:
                e_hole.append(float(np.sum(diff2 * valid * m) / denom_h))

        if lpips_metric is not None:
            tlp_all.append(lpips_metric(pred_t, warped_pred_tp1))
            if mpath is not None and Path(mpath).exists():
                m = load_mask(Path(mpath), size=pix)
                ys, xs = np.where(m > 0.5)
                if len(xs) > 0:
                    y0, y1 = ys.min(), ys.max()
                    x0, x1 = xs.min(), xs.max()
                    pad = 8
                    y0 = max(0, y0 - pad); y1 = min(pred_t.shape[0]-1, y1 + pad)
                    x0 = max(0, x0 - pad); x1 = min(pred_t.shape[1]-1, x1 + pad)
                    tlp_hole.append(lpips_metric(pred_t[y0:y1+1, x0:x1+1, :],
                                                 warped_pred_tp1[y0:y1+1, x0:x1+1, :]))

    return {
        "Ewarp_all": float(np.mean(e_all)) if len(e_all) else float("nan"),
        "Ewarp_hole": float(np.mean(e_hole)) if len(e_hole) else float("nan"),
        "tLPIPS_all": float(np.mean(tlp_all)) if len(tlp_all) else float("nan"),
        "tLPIPS_hole": float(np.mean(tlp_hole)) if len(tlp_hole) else float("nan"),
    }


def collect_masks_index_aligned(mask_root: Path, seq_name: str) -> List[Path]:
    """
    Prefer mask_root/seq_name/* if exists; otherwise search masks under mask_root that contain seq_name in path;
    otherwise take all masks under mask_root.
    Always natural-sorted.
    """
    cand_dir = mask_root / seq_name
    if cand_dir.exists():
        return list_images(cand_dir)

    allm = list_images(mask_root)
    filt = [p for p in allm if seq_name in p.as_posix().split("/")]
    if len(filt) > 0:
        filt.sort(key=natural_key)
        return filt

    return allm




def write_mask_pairing_log(
    out_dir: Path,
    pred_frames: List[Path],
    gt_frames: List[Path],
    mask_frames: List[Optional[Path]],
) -> Tuple[Path, Path]:
    """
    Write a simple index-aligned pairing log for sanity-checking.
    Each row records: idx, pred_name, gt_name, mask_name, mask_path, mask_exists
    """
    out_txt = out_dir / "mask_pairing_log.txt"
    out_csv = out_dir / "mask_pairing_log.csv"

    # txt
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("idx\tpred\tgt\tmask\tmask_path\tmask_exists\n")
        for i, (p, g, m) in enumerate(zip(pred_frames, gt_frames, mask_frames)):
            m_name = m.name if m is not None else "None"
            m_path = str(m) if m is not None else ""
            m_ok = (m is not None and Path(m).exists())
            f.write(f"{i}\t{p.name}\t{g.name}\t{m_name}\t{m_path}\t{int(m_ok)}\n")

    # csv
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "pred", "gt", "mask", "mask_path", "mask_exists"])
        for i, (p, g, m) in enumerate(zip(pred_frames, gt_frames, mask_frames)):
            m_name = m.name if m is not None else "None"
            m_path = str(m) if m is not None else ""
            m_ok = (m is not None and Path(m).exists())
            w.writerow([i, p.name, g.name, m_name, m_path, int(m_ok)])

    return out_txt, out_csv
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default="")
    parser.add_argument("--pix", type=int, default=256)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--seq_name", type=str, default="",
                        help="mask 子目录名（默认使用 gt_path 的 basename，例如 fe30a8）")
    parser.add_argument("--vfid_backbone", type=str, default="r3d_18",
                        choices=["r3d_18", "mc3_18", "r2plus1d_18"])
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=8)
    parser.add_argument("--flow_source", type=str, default="gt", choices=["gt", "pred"])
    # ---- metric switches (all disabled by default) ----
    parser.add_argument("--PSNR_all", action="store_true", default=True, help="Enable PSNR on full frame")
    parser.add_argument("--PSNR_hole", action="store_true", default=True, help="Enable PSNR on hole region (requires mask)")
    parser.add_argument("--SSIM_all", action="store_true", default=False, help="Enable SSIM on full frame")
    parser.add_argument("--SSIM_hole", action="store_true", default=False, help="Enable SSIM on hole region (requires mask)")
    parser.add_argument("--LPIPS", action="store_true", default=True, help="Enable LPIPS (per-frame perceptual distance)")
    parser.add_argument("--FID", action="store_true", default=False, help="Enable FID (image distribution distance)")
    parser.add_argument("--VFID", action="store_true", default=True, help="Enable VFID proxy (video distribution distance)")
    parser.add_argument("--Ewarp_all", action="store_true", default=True, help="Enable Ewarp on full frame (temporal, requires optical flow)")
    parser.add_argument("--Ewarp_hole", action="store_true", default=False, help="Enable Ewarp on hole region (temporal, requires mask & optical flow)")
    parser.add_argument("--tLPIPS_all", action="store_true", default=False, help="Enable temporal LPIPS on full frame (temporal, requires optical flow)")
    parser.add_argument("--tLPIPS_hole", action="store_true", default=False, help="Enable temporal LPIPS on hole region (temporal, requires mask & optical flow)")

    parser.add_argument("--out", type=str, default="metrics_out")
    
    parser.add_argument(
    "--mask_pairing_log",
    action="store_true",
    default=False,
    help="Enable writing mask–frame pairing log (default: off)"
)
    
    args = parser.parse_args()

    pred_root = Path(args.pre_path)
    gt_root = Path(args.gt_path)
    mask_root = Path(args.mask_path) if args.mask_path else None

    assert pred_root.exists(), f"pre_path not found: {pred_root}"
    assert gt_root.exists(), f"gt_path not found: {gt_root}"
    if mask_root is not None:
        assert mask_root.exists(), f"mask_path not found: {mask_root}"



    device = torch.device(args.device)
    pix = args.pix if args.resize else None

    # seq name for outputs / mask folder hint
    seq_name = args.seq_name.strip() if args.seq_name.strip() else gt_root.name

    # 1) pred / gt：按自然排序 index 对齐
    pred_frames = list_images(pred_root)
    gt_frames = list_images(gt_root)
    if len(pred_frames) == 0 or len(gt_frames) == 0:
        raise RuntimeError("pred 或 gt 目录下没有找到图片")

    N = min(len(pred_frames), len(gt_frames))
    if len(pred_frames) != len(gt_frames):
        print(f"[WARN] pred={len(pred_frames)} gt={len(gt_frames)} -> will index-align first N={N} frames")

    pred_frames = pred_frames[:N]
    gt_frames = gt_frames[:N]

    # 2) mask：按自然排序 index 对齐
    mask_frames: List[Optional[Path]] = [None] * N
    if mask_root is not None:
        # seq_name is already set above
        masks = collect_masks_index_aligned(mask_root, seq_name)
        if len(masks) == 0:
            print("[WARN] mask_path 下未找到任何 mask，hole 指标将为 NaN")
        else:
            if len(masks) < N:
                print(f"[WARN] masks={len(masks)} < frames={N} -> remaining frames have no mask")
            if len(masks) > N:
                print(f"[INFO] masks={len(masks)} > frames={N} -> take first N masks by natural order")
            for i in range(min(N, len(masks))):
                mask_frames[i] = masks[i]
            print(f"[INFO] mask matched by index. seq_name={seq_name}. used_masks={min(N,len(masks))}")
            
    out_dir = Path(args.out + '/' + seq_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- mask pairing sanity log (index-aligned) ---
    try:
        if args.mask_pairing_log:
            log_txt, log_csv = write_mask_pairing_log(out_dir, pred_frames, gt_frames, mask_frames)
            print(f"[INFO] Wrote mask pairing log: {log_txt}")
    except Exception as e:
        print(f"[WARN] Failed to write mask pairing log: {e}")


    # LPIPS (only initialize when requested)
    lpips_metric = None
    need_lpips = bool(args.LPIPS or args.tLPIPS_all or args.tLPIPS_hole)
    if need_lpips:
        if lpips is not None:
            lpips_metric = LPIPSMetric(device=device, net="alex")
        else:
            print("[WARN] lpips not installed; LPIPS/tLPIPS will be skipped.")

    # Frame metrics (only compute requested ones)
    psnr_all_list, psnr_hole_list = [], []
    ssim_all_list, ssim_hole_list = [], []
    lpips_list = []

    need_frame_metrics = bool(args.PSNR_all or args.PSNR_hole or args.SSIM_all or args.SSIM_hole or args.LPIPS)
    if need_frame_metrics:
        for p, g, m in tqdm(list(zip(pred_frames, gt_frames, mask_frames)), desc="Frame metrics", total=N):
            pred = pil_to_np_rgb01(Image.open(p), size=pix)
            gt = pil_to_np_rgb01(Image.open(g), size=pix)

            if args.PSNR_all:
                psnr_all_list.append(float(peak_signal_noise_ratio(gt, pred, data_range=1.0)))
            if args.SSIM_all:
                ssim_all_list.append(float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0)))

            if (args.PSNR_hole or args.SSIM_hole) and (m is not None and Path(m).exists()):
                mask = load_mask(Path(m), size=pix)
                if args.PSNR_hole:
                    psnr_hole_list.append(psnr_hole(pred, gt, mask))
                if args.SSIM_hole:
                    ssim_hole_list.append(ssim_hole_bbox(pred, gt, mask))
            else:
                # keep NaN placeholders so nanmean behaves as expected
                if args.PSNR_hole:
                    psnr_hole_list.append(float("nan"))
                if args.SSIM_hole:
                    ssim_hole_list.append(float("nan"))

            if args.LPIPS and (lpips_metric is not None):
                lpips_list.append(lpips_metric(pred, gt))
    # FID
    fid = float("nan")
    if args.FID:
        try:
            inc = InceptionPool3(device=device)
            fid = fid_features(pred_frames, gt_frames, pix, inc, device, args.batch_size)
        except Exception as e:
            print(f"[WARN] FID failed: {e}")
    # VFID proxy
    vfid = float("nan")
    if args.VFID:
        try:
            vb = VideoBackbone(args.vfid_backbone, device=device)
            vfid = compute_vfid_proxy(
                pred_frames, gt_frames, vb, device, pix,
                clip_len=args.clip_len, stride=args.clip_stride, batch_size=8
            )
        except Exception as e:
            print(f"[WARN] VFID failed: {e}")
    # Temporal metrics
    temporal = {
        "Ewarp_all": float("nan"),
        "Ewarp_hole": float("nan"),
        "tLPIPS_all": float("nan"),
        "tLPIPS_hole": float("nan"),
    }
    need_temporal = bool(args.Ewarp_all or args.Ewarp_hole or args.tLPIPS_all or args.tLPIPS_hole)
    if need_temporal:
        if cv2 is None:
            print("[WARN] opencv (cv2) not installed; Ewarp/tLPIPS will be skipped.")
        else:
            temporal_full = compute_ewarp_and_tlpips(
                pred_frames, gt_frames, mask_frames, pix, lpips_metric, flow_source=args.flow_source
            )
            if args.Ewarp_all:
                temporal["Ewarp_all"] = temporal_full.get("Ewarp_all", float("nan"))
            if args.Ewarp_hole:
                temporal["Ewarp_hole"] = temporal_full.get("Ewarp_hole", float("nan"))
            if args.tLPIPS_all:
                temporal["tLPIPS_all"] = temporal_full.get("tLPIPS_all", float("nan"))
            if args.tLPIPS_hole:
                temporal["tLPIPS_hole"] = temporal_full.get("tLPIPS_hole", float("nan"))

    # ---------------- results (ONLY include enabled metrics; no placeholders) ----------------
    enabled_metrics: List[str] = []
    if args.PSNR_all: enabled_metrics.append("PSNR_all")
    if args.PSNR_hole: enabled_metrics.append("PSNR_hole")
    if args.SSIM_all: enabled_metrics.append("SSIM_all")
    if args.SSIM_hole: enabled_metrics.append("SSIM_hole")
    if args.LPIPS: enabled_metrics.append("LPIPS")
    if args.FID: enabled_metrics.append("FID")
    if args.VFID: enabled_metrics.append("VFID_proxy")
    if args.Ewarp_all: enabled_metrics.append("Ewarp_all")
    if args.Ewarp_hole: enabled_metrics.append("Ewarp_hole")
    if args.tLPIPS_all: enabled_metrics.append("tLPIPS_all")
    if args.tLPIPS_hole: enabled_metrics.append("tLPIPS_hole")

    results: Dict[str, object] = {
        "count_frames": N,
        "enabled_metrics": enabled_metrics,
    }

    # frame metrics
    if args.PSNR_all and len(psnr_all_list):
        results["PSNR_all"] = float(np.nanmean(psnr_all_list))
    if args.PSNR_hole and len(psnr_hole_list):
        results["PSNR_hole"] = float(np.nanmean(psnr_hole_list))
    if args.SSIM_all and len(ssim_all_list):
        results["SSIM_all"] = float(np.nanmean(ssim_all_list))
    if args.SSIM_hole and len(ssim_hole_list):
        results["SSIM_hole"] = float(np.nanmean(ssim_hole_list))
    if args.LPIPS and len(lpips_list):
        results["LPIPS"] = float(np.nanmean(lpips_list))

    # distribution metrics
    if args.FID:
        results["FID"] = float(fid)
    if args.VFID:
        results["VFID_proxy"] = float(vfid)
        results["VFID_backbone"] = args.vfid_backbone

    # temporal metrics
    if need_temporal:
        results["flow_source"] = args.flow_source
        if args.Ewarp_all:
            results["Ewarp_all"] = float(temporal.get("Ewarp_all", float("nan")))
        if args.Ewarp_hole:
            results["Ewarp_hole"] = float(temporal.get("Ewarp_hole", float("nan")))
        if args.tLPIPS_all:
            results["tLPIPS_all"] = float(temporal.get("tLPIPS_all", float("nan")))
        if args.tLPIPS_hole:
            results["tLPIPS_hole"] = float(temporal.get("tLPIPS_hole", float("nan")))


    print("\n========== Metrics ==========")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:>15s}: {v:.6f}")
        else:
            print(f"{k:>15s}: {v}")
    print("=============================\n")

    # Write metrics logs ONLY when at least one metric is enabled.
    if len(results.get("enabled_metrics", [])) == 0:
        print("[INFO] No metrics enabled -> skip writing metrics.json / metrics.csv")
    else:
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        import csv
        # --- write metrics.csv (exclude non-metric meta fields) ---
        csv_exclude = {"VFID_backbone", "flow_source"}

        csv_results = {k: v for k, v in results.items() if k not in csv_exclude}

        # 如果剔除后为空，就不写 csv（避免只写表头或空文件）
        if len(csv_results) > 0:
            with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_results.keys()))
                writer.writeheader()
                writer.writerow(csv_results)

        print(f"Saved: {out_dir / 'metrics.json'}")
        print(f"Saved: {out_dir / 'metrics.csv'}")



if __name__ == "__main__":
    main()

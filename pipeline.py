"""YOLO26 detect -> ByteTrack -> Dense optical flow -> OViF -> window feature vector.

Change vs purana version: speed aur min_dist ab object ke size ke hisaab se normalize hain
(perspective / camera angle ka asar kam). Feature ke naam aur tadaad (20) wahi hain.

NAYA (sirf drawing ke liye, model ka decision/features bilkul nahi badle):
_features() ab ye bhi batata hai ke window ke max_iou wale asal pair (tid1, tid2)
kaun se the -- yani jin 2 cars ka overlap waqai alert signal mein contribute kar raha
tha. Agar koi pair MIN_IOU_FOR_HIGHLIGHT se zyada overlap nahi karta to pair None hota
hai. Ye sirf detect_all.py mein red-box highlight ke liye use hota hai.
"""
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO
import config as C

FEATURE_NAMES = (
    ["n_tracks", "max_speed", "mean_speed", "max_decel", "max_speed_drop",
     "max_dir_change", "max_iou", "min_dist", "flow_mean", "flow_std",
     "flow_peak", "flow_change"]
    + [f"ovif_{i}" for i in range(C.OVIF_BINS)]
)

NO_PAIR_DIST = 10.0  # min_dist jab koi do objects ek frame mein na hon (size units)


def load_model():
    try:
        return YOLO(C.MODEL)
    except Exception as e:
        print(f"[warn] {C.MODEL} load nahi hua ({e}); {C.FALLBACK_MODEL} use ho raha hai")
        return YOLO(C.FALLBACK_MODEL)


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def obj_size(b):
    """Object ka size = sqrt(w*h) (normalized coords mein)."""
    return float(np.sqrt(max(b[2] * b[3], 1e-8)))


MIN_IOU_FOR_HIGHLIGHT = 0.02  # below this, no two boxes actually overlap -> nothing to highlight


class Analyzer:
    def __init__(self, model=None):
        self.model = model or load_model()
        self.frames = deque(maxlen=C.WINDOW)
        self.count = 0
        self.prev_gray = None
        self.prev_mag = None

    def _flow(self, frame):
        g = cv2.cvtColor(cv2.resize(frame, C.FLOW_SIZE), cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = g
            self.prev_mag = np.zeros(g.shape, np.float32)
            return 0.0, 0.0, 0.0, np.zeros(C.OVIF_BINS)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # OViF: orientation histogram of pixels with significant magnitude change
        diff = np.abs(mag - self.prev_mag)
        mask = diff > diff.mean()
        bins = (ang / (2 * np.pi) * C.OVIF_BINS).astype(int) % C.OVIF_BINS
        hist = np.bincount(bins[mask], minlength=C.OVIF_BINS).astype(float)
        stats = (float(mag.mean()), float(mag.std()), float(diff.mean()))
        self.prev_gray, self.prev_mag = g, mag
        return stats[0], stats[1], stats[2], hist

    def step(self, frame):
        """Returns (feature_vector or None, list of (id, x1,y1,x2,y2) pixel boxes,
        collision_pair). collision_pair is the (tid1, tid2) that produced the window's
        max_iou -- i.e. the actual pair driving the accident signal -- and is only set
        on the same frames where a feature vector is computed (every STRIDE frames once
        the window is full). It is None the rest of the time, and None whenever no pair
        in the window actually overlapped (max_iou below MIN_IOU_FOR_HIGHLIGHT) -- so
        two cars merely driving near each other in adjacent lanes are never marked."""
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                             classes=C.CLASSES, conf=C.CONF, imgsz=C.IMGSZ, verbose=False)[0]
        tracks, draw = {}, []
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            xywhn = r.boxes.xywhn.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for i, tid in enumerate(ids):
                tracks[tid] = tuple(xywhn[i])
                draw.append((tid, *map(int, xyxy[i])))
        fm, fs, fd, hist = self._flow(frame)
        self.frames.append({"tracks": tracks, "fm": fm, "fs": fs, "fd": fd, "hist": hist})
        self.count += 1
        if len(self.frames) == C.WINDOW and self.count % C.STRIDE == 0:
            vec, pair = self._features()
            return vec, draw, pair
        return None, draw, None

    def _features(self):
        series = {}
        for fi, f in enumerate(self.frames):
            for tid, b in f["tracks"].items():
                series.setdefault(tid, []).append((fi, *b))
        speeds_all, decels, drops, dirs = [], [], [], []
        for s in series.values():
            if len(s) < 6:
                continue
            a = np.array(s)
            dt = np.maximum(np.diff(a[:, 0]), 1)
            # size-normalized velocity: "object size per frame"
            size = float(np.mean(np.sqrt(np.maximum(a[:, 3] * a[:, 4], 1e-8))))
            vel = np.diff(a[:, 1:3], axis=0) / dt[:, None] / max(size, 1e-3)
            sp = np.linalg.norm(vel, axis=1)
            speeds_all.append(sp)
            if len(sp) > 1:
                decels.append(float(-np.min(np.diff(sp))))
            k = max(len(sp) // 3, 1)
            first, last = sp[:k].mean(), sp[-k:].mean()
            if first > 1e-3:
                drops.append(float((first - last) / first))
            v1, v2 = vel[:len(vel)//2].mean(0), vel[len(vel)//2:].mean(0)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-4 and n2 > 1e-4:
                dirs.append(float(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1))))
        max_iou, min_dist, iou_pair = 0.0, NO_PAIR_DIST, None
        for f in self.frames:
            items = list(f["tracks"].items())
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    tid1, b1 = items[i]
                    tid2, b2 = items[j]
                    iou = iou_xywh(b1, b2)
                    if iou > max_iou:
                        max_iou, iou_pair = iou, (tid1, tid2)
                    d = np.hypot(b1[0] - b2[0], b1[1] - b2[1])
                    # faasla objects ke average size ke units mein
                    ref = (obj_size(b1) + obj_size(b2)) / 2
                    min_dist = min(min_dist, float(d / max(ref, 1e-3)))
        if max_iou < MIN_IOU_FOR_HIGHLIGHT:
            iou_pair = None  # koi 2 boxes asal mein overlap nahi hue -- highlight karne ko kuch nahi
        allsp = np.concatenate(speeds_all) if speeds_all else np.array([0.0])
        fm = np.array([f["fm"] for f in self.frames])
        fd = np.array([f["fd"] for f in self.frames])
        hist = np.sum([f["hist"] for f in self.frames], axis=0)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        vec = [len(series), allsp.max(), allsp.mean(), max(decels, default=0.0),
               max(drops, default=0.0), max(dirs, default=0.0), max_iou, min_dist,
               fm.mean(), fm.std(), fm.max(), fd.max(), *hist]
        return np.array(vec, dtype=np.float32), iou_pair
    
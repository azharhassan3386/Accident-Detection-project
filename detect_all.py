"""Folder ki saari videos ek ke baad ek chala kar accident detection dikhata hai.

Folder mein ya to accident/ aur normal/ sub-folders hon (jaise data/ ya test/),
ya seedha videos hon.

Keys (window par):  n = agli video   SPACE = pause/resume   q = band
                    f = tez   s = slow   (live mode: skip +1/-1,  prerender mode: playback speed)
                    r = replay (prerender mode)

Tareeqe:
  1) LIVE (default): har frame ek ek karke analyse hoti hai aur foran dikhti hai.
     CPU par ~2 FPS = slow motion. --skip 2 tez karta hai, lekin accuracy gir sakti hai!
  2) --prerender: pehli baar har video har frame par analyse hoti hai (EXACT detection),
     nateeja cache/ folder mein SAVE hota hai, phir video normal speed par chalti hai.
     Dobara chalane par analysis nahi hota -- video FORAN normal speed par chalti hai.
  3) --build-cache: saari videos ka analysis (bina window ke) ek saath kar ke save kar do.
     Isay chala kar chhor do (chai pi lo), phir --prerender se sab foran chalti hain.

Options:
  --imgsz 960     YOLO input size (default 960; 0 = config.py wali value). Chhota = tez.
  --skip 1        (sirf live mode) har N-wi frame analyse karo.
  --speed 1.0     prerender playback speed (0.25 ... 4).
  --start 3       3rd video se shuru karo.
  --recompute     purana cache ignore karke dobara analyse karo.
Cache tab khud purana ho jata hai jab model.json / pipeline.py / config.py badal jayein ya imgsz alag ho.

jab ACCIDENT DETECTED ho, to sirf collision wali 2 cars (closest/most-overlapping
pair us frame mein) red box mein highlight hoti hain -- baaki sab cars green hi rehti hain,
bilkul pehle ki tarah."""
import argparse, glob, os, pickle, time
from types import SimpleNamespace
import cv2
from xgboost import XGBClassifier
import config as C
from pipeline import Analyzer, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
WIN = "YoFlow-26 (n = next, space = pause, q = quit)"

p = argparse.ArgumentParser()
p.add_argument("--folder", default="data")
p.add_argument("--model", default="model.json")
p.add_argument("--only", choices=["accident", "normal"], default=None,
               help="sirf ek qism ki videos chalayein")
p.add_argument("--maxw", type=int, default=960, help="window ki max chorai (pixels)")
p.add_argument("--imgsz", type=int, default=0,
               help="YOLO input size (default 960). Chhota = tez, par door ki gariyan kam pakri jati hain. "
                    "0 = config.py wali value (1280) use karo")
p.add_argument("--skip", type=int, default=1,
               help="(live mode) har N-wi frame analyse + dikhao. 1 = har frame (model ke training jaisa)")
p.add_argument("--prerender", action="store_true",
               help="har frame analyse (exact), nateeja save, phir normal speed par chalao")
p.add_argument("--build-cache", action="store_true",
               help="saari videos ka analysis bina window ke karke save karo, phir band")
p.add_argument("--recompute", action="store_true", help="purana cache ignore karo")
p.add_argument("--cache", default=os.path.join(HERE, "cache"), help="cache folder")
p.add_argument("--speed", type=float, default=1.0, help="prerender playback speed (0.25 ... 4)")
p.add_argument("--start", type=int, default=1, help="is number wali video se shuru karo (1 = pehli)")
p.add_argument("--save", action="store_true",
                help="annotated video results/ folder mein save karo (window nahi dikhegi)")
a = p.parse_args()
if a.imgsz:
    C.IMGSZ = a.imgsz          # pipeline.py yahi value use karta hai
skip = max(1, a.skip)
speed_i = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - a.speed))
proc_fps = 0.0


def videos_in(folder):
    return [f for f in sorted(glob.glob(os.path.join(folder, "*")))
            if os.path.splitext(f)[1].lower() in VIDEO_EXT]


items = []  # (label, path)
subs = [s for s in ("accident", "normal") if os.path.isdir(os.path.join(a.folder, s))]
if subs:
    for s in subs:
        if a.only and s != a.only:
            continue
        items += [(s, v) for v in videos_in(os.path.join(a.folder, s))]
else:
    items = [("", v) for v in videos_in(a.folder)]
if not items:
    raise SystemExit(f"'{a.folder}' mein koi video nahi mili")

clf = XGBClassifier()
clf.load_model(a.model)
yolo = load_model()
mode_txt = ("BUILD-CACHE" if a.build_cache else
            "PRERENDER (exact analyse + cache, phir normal speed playback)" if a.prerender else f"LIVE skip={skip}")
print(f"{len(items)} videos. IMGSZ={C.IMGSZ}  {mode_txt}. Keys: n = agli, SPACE = pause, f/s = tez/slow, q = band")


class Run:
    """Ek video ka alert-state (probability, streak, hold, red pair)."""

    def __init__(self, fps):
        self.fps = fps
        self.prob, self.streak, self.alert, self.hold, self.maxp = 0.0, 0, False, 0, 0.0
        self.last_pair = None  # (tid1, tid2) jo abhi red honi chahiye -- hold ke dauran yaad rakhte hain

    def feed(self, vec, pair, step):
        if vec is not None:
            self.prob = float(clf.predict_proba(vec.reshape(1, -1))[0, 1])
            self.maxp = max(self.maxp, self.prob)
            self.streak = self.streak + 1 if self.prob >= C.ALERT_THRESHOLD else 0
            if self.streak >= C.ALERT_CONSECUTIVE:
                self.alert, self.hold = True, max(1, int(self.fps * 3 / step))
        if self.hold > 0:
            self.hold -= 1
        else:
            self.alert = False
        if pair is not None:
            self.last_pair = pair  # sirf jab naya pair mile tab update, tracker glitch mein purana yaad rahega
        if not self.alert:
            self.last_pair = None


def annotate(frame, boxes, run, k, name, bottom_text=""):
    H, W = frame.shape[:2]
    for tid, x1, y1, x2, y2 in boxes:
        is_colliding = run.alert and run.last_pair is not None and tid in run.last_pair
        color = (0, 0, 255) if is_colliding else (0, 255, 0)
        thick = 3 if is_colliding else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, f"#{tid}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"[{k}/{len(items)}] {name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"accident prob: {run.prob:.2f}  (max {run.maxp:.2f})", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if run.alert:
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 8)
        cv2.putText(frame, "ACCIDENT DETECTED", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    if bottom_text:
        cv2.putText(frame, bottom_text, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if W > a.maxw:
        s = a.maxw / W
        frame = cv2.resize(frame, (a.maxw, int(H * s)))
    return frame


def get_key(delay=1):
    key = cv2.waitKey(delay) & 0xFF
    if 65 <= key <= 90:                    # Caps Lock safe
        key += 32
    return key


def open_video(path, name):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[skip] {name}: khuli nahi")
        return None, 0
    return cap, (cap.get(cv2.CAP_PROP_FPS) or 25)


def fmt_eta(el, n, total):
    if not total or not n:
        return ""
    eta = int(el / n * max(total - n, 0))
    return f"  ETA {eta // 60}:{eta % 60:02d}"


# ---------------------------------------------------------------- cache
def cache_file(label, path):
    base = f"{label}_{os.path.basename(path)}_{C.IMGSZ}.pkl".lstrip("_")
    return os.path.join(a.cache, base)


def load_cache(cp):
    """(data, fps) ya (None, 0) agar cache nahi / purana / recompute."""
    if a.recompute or not os.path.exists(cp):
        return None, 0
    t = os.path.getmtime(cp)
    for dep in (a.model, os.path.join(HERE, "pipeline.py"), os.path.join(HERE, "config.py")):
        if os.path.exists(dep) and os.path.getmtime(dep) > t:
            return None, 0
    try:
        with open(cp, "rb") as f:
            d = pickle.load(f)
        return d["data"], d["fps"]
    except Exception:
        return None, 0


def save_cache(cp, data, fps):
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump({"data": data, "fps": fps}, f)


# ---------------------------------------------------------------- analysis (har frame, exact)
def analyse(path, name, k, show=True):
    """Returns (data, fps, status). data = [(boxes, prob, maxp, alert, last_pair), ...] har frame ke liye.
    status: "ok" / "skip" (n dabaya) / "quit" (q dabaya)."""
    cap, fps = open_video(path, name)
    if cap is None:
        return None, 0, "skip"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    an = Analyzer(yolo)
    yolo.predictor = None  # tracker reset
    run = Run(fps)
    data, n, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vec, boxes, pair = an.step(frame)
        run.feed(vec, pair, 1)
        data.append((list(boxes), run.prob, run.maxp, run.alert, run.last_pair))
        n += 1
        if show:
            if n % 3 == 0:                                         # progress dikhane ke liye kabhi kabhi frame
                pct = f"{100 * n // total}%" if total else f"{n} frames"
                disp = annotate(frame, boxes, run, k, name)
                cv2.putText(disp, f"Analysing {pct}{fmt_eta(time.time() - t0, n, total)}   (n = skip video, q = quit)",
                            (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(WIN, disp)
            key = get_key()
            if key == ord("q"):
                cap.release()
                return None, fps, "quit"
            if key == ord("n"):
                cap.release()
                print(f"{name}: skip kar di")
                return None, fps, "skip"
        elif n % 25 == 0:
            pct = f"{100 * n // total}%" if total else f"{n} frames"
            print(f"\r  [{k}/{len(items)}] {name}: {pct}{fmt_eta(time.time() - t0, n, total)}      ", end="", flush=True)
    cap.release()
    if not show:
        print()
    return data, fps, "ok"
def pair_stats(a, b):
    """a, b = (tid, x1, y1, x2, y2). Returns (iou, size_ratio)."""
    a1 = max((a[3] - a[1]) * (a[4] - a[2]), 1)
    a2 = max((b[3] - b[1]) * (b[4] - b[2]), 1)
    iw = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ih = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = iw * ih
    iou = inter / (a1 + a2 - inter)
    s1, s2 = a1 ** 0.5, a2 ** 0.5
    return iou, min(s1, s2) / max(s1, s2)


def best_pair(data, lo, hi, min_iou=0.03, min_ratio=0.45, max_overlap_frac=0.8, min_together=5):
    """lo..hi frames ke cached boxes mein se asli takkar wali 2 garian dhundta hai."""
    together, overlap, peak = {}, {}, {}
    for f in range(lo, hi + 1):
        boxes = data[f][0]
        for x in range(len(boxes)):
            for y in range(x + 1, len(boxes)):
                a, b = boxes[x], boxes[y]
                key = (min(a[0], b[0]), max(a[0], b[0]))
                together[key] = together.get(key, 0) + 1
                iou, ratio = pair_stats(a, b)
                if iou >= min_iou and ratio >= min_ratio:
                    overlap[key] = overlap.get(key, 0) + 1
                    peak[key] = max(peak.get(key, 0.0), iou)
    # har waqt overlap mein rehne wali jodi (rider+bike, peechay chalti gari) takkar nahi hoti
    cands = [k for k in overlap
             if overlap[k] >= 2 and together[k] >= min_together
             and overlap[k] / together[k] <= max_overlap_frac]
    if not cands:
        return None
    return max(cands, key=lambda k: peak[k] * min(overlap[k], 8))


def refine_pairs(data, fps):
    """Har alert ke liye ek pakki (fixed) pair chunta hai. Returns {frame_index: (tid1, tid2)}."""
    n, fixed, i = len(data), {}, 0
    while i < n:
        if not data[i][3]:
            i += 1
            continue
        s = i
        while i < n and data[i][3]:
            i += 1
        e = i - 1
        lo = max(0, s - int(fps * 2.5))
        hi = min(n - 1, s + int(fps * 2))
        pair = best_pair(data, lo, hi)
        if pair is not None:
            for j in range(s, e + 1):
                fixed[j] = pair
    return fixed
def pair_stats(a, b):
    """a, b = (tid, x1, y1, x2, y2). Returns (iou, size_ratio)."""
    a1 = max((a[3] - a[1]) * (a[4] - a[2]), 1)
    a2 = max((b[3] - b[1]) * (b[4] - b[2]), 1)
    iw = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ih = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = iw * ih
    s1, s2 = a1 ** 0.5, a2 ** 0.5
    return inter / (a1 + a2 - inter), min(s1, s2) / max(s1, s2)


def _center_tracks(data, lo, hi):
    """tid -> {frame: (cx, cy, size)}"""
    tr = {}
    for f in range(lo, hi + 1):
        for tid, x1, y1, x2, y2 in data[f][0]:
            tr.setdefault(tid, {})[f] = ((x1 + x2) / 2, (y1 + y2) / 2,
                                         max(((x2 - x1) * (y2 - y1)) ** 0.5, 1.0))
    return tr


def _velocity(track, f0, f1):
    """Net displacement se raftaar (size units/frame, dx, dy). Jitter average ho jata hai."""
    fr = [f for f in range(f0, f1 + 1) if f in track]
    if len(fr) < 4 or fr[-1] - fr[0] < 4:
        return None
    a, b = track[fr[0]], track[fr[-1]]
    span, size = fr[-1] - fr[0], max(a[2], b[2], 1.0)
    return ((b[0] - a[0]) / span / size, (b[1] - a[1]) / span / size)


def _jolt(tr, key, c, k=10):
    """Takkar ke baad raftaar/rukh mein achanak tabdeeli (0 = koi nahi, 1 = bohat zyada)."""
    best = 0.0
    for tid in key:
        vb = _velocity(tr[tid], c - k, c - 1)
        va = _velocity(tr[tid], c + 3, c + k + 2)
        if vb is None or va is None:
            continue
        top = max((vb[0] ** 2 + vb[1] ** 2) ** 0.5, (va[0] ** 2 + va[1] ** 2) ** 0.5)
        if top > 0.004:
            change = ((vb[0] - va[0]) ** 2 + (vb[1] - va[1]) ** 2) ** 0.5 / top
            best = max(best, min(change, 1.0))
    return best


def _collision_start(frames, peak_frame, gap=3):
    """Peak overlap wali run ka pehla frame (jab 2 garian pehli baar takrain)."""
    i = frames.index(peak_frame)
    while i > 0 and frames[i] - frames[i - 1] <= gap:
        i -= 1
    return frames[i]


def best_pair(data, lo, hi, min_iou=0.03, min_ratio=0.45, min_apart=3, min_together=5):
    """lo..hi frames mein asli takkar: Returns ((tid1, tid2), collision_start_frame) ya None."""
    together, overlap, peak, peak_f = {}, {}, {}, {}
    for f in range(lo, hi + 1):
        boxes = data[f][0]
        for x in range(len(boxes)):
            for y in range(x + 1, len(boxes)):
                a, b = boxes[x], boxes[y]
                key = (min(a[0], b[0]), max(a[0], b[0]))
                together.setdefault(key, []).append(f)
                iou, ratio = pair_stats(a, b)
                if iou >= min_iou and ratio >= min_ratio:
                    overlap.setdefault(key, []).append(f)
                    if iou > peak.get(key, 0.0):
                        peak[key], peak_f[key] = iou, f
    tr = _center_tracks(data, lo, hi)
    best, best_score = None, 0.0
    for key, frames in overlap.items():
        if len(frames) < 2 or len(together[key]) < min_together:
            continue
        start = _collision_start(frames, peak_f[key])
        ov = set(frames)
        apart = sum(1 for f in together[key] if f < start and f not in ov)
        if apart < min_apart:      # shuru se hi sath chipki (rider+bike, peechay chalti gari) = takkar nahi
            continue
        score = peak[key] * min(len(frames), 8) * (0.3 + _jolt(tr, key, start))
        if score > best_score:
            best, best_score = (key, start), score
    return best


def refine_pairs(data, fps):
    """Har alert ke liye {frame_index: (tid1, tid2)}. Red takkar ke pehle frame se shuru hota hai."""
    n, fixed, i = len(data), {}, 0
    while i < n:
        if not data[i][3]:
            i += 1
            continue
        s = i
        while i < n and data[i][3]:
            i += 1
        e = i - 1
        found = best_pair(data, max(0, s - int(fps * 2.5)), min(n - 1, s + int(fps * 2)))
        if found:
            pair, c = found
            for j in range(min(c, s), e + 1):
                fixed[j] = pair
    return fixed
# ---------------------------------------------------------------- playback (cache se, normal speed)
def play_cached(path, name, k, data, fps):
    """Original video ko cache ke boxes ke saath normal speed par chalata hai. Returns True agar 'q' dabaya."""
    global speed_i
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[skip] {name}: khuli nahi")
        return False
    fps = max(fps, 25)
    fixed = refine_pairs(data, fps)
    i, paused, quit_all = 0, False, False
    while True:
        t_loop = time.time()
        if not paused:
            ok, frame = cap.read()
            if not ok or i >= len(data):
                break
            boxes, prob, maxp, alert, pair = data[i]
            if i in fixed:
                alert, pair = True, fixed[i]
            st = SimpleNamespace(prob=prob, maxp=maxp, alert=alert, last_pair=pair)
            cv2.imshow(WIN, annotate(frame, boxes, st, k, name,
                                     f"play x{SPEEDS[speed_i]:g}   (f = tez, s = slow, r = replay, space = pause)"))
            i += 1
        delay = max(1, int(1000.0 / (fps * SPEEDS[speed_i]) - (time.time() - t_loop) * 1000.0))
        key = get_key(delay)
        if key == ord("q"):
            quit_all = True
            break
        if key == ord("n"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("f"):
            speed_i = min(speed_i + 1, len(SPEEDS) - 1)
        if key == ord("s"):
            speed_i = max(speed_i - 1, 0)
        if key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            i = 0
    cap.release()
    return quit_all


def export_cached(path, name, k, data, fps):
    """Cache ke boxes ke saath annotated video results/ folder mein save karta hai."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    out_path = os.path.join(HERE, "results", name.replace(os.sep, "_").rsplit(".", 1)[0] + "_result.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fixed = refine_pairs(data, fps)
    writer, i = None, 0
    while i < len(data):
        ok, frame = cap.read()
        if not ok:
            break
        boxes, prob, maxp, alert, pair = data[i]
        if i in fixed:
            alert, pair = True, fixed[i]
        st = SimpleNamespace(prob=prob, maxp=maxp, alert=alert, last_pair=pair)
        out = annotate(frame, boxes, st, k, name)
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(out)
        i += 1
    cap.release()
    if writer:
        writer.release()
    print(f"{name}: saved -> {out_path}")

def play_prerender(path, name, k):
    cp = cache_file(*items[k - 1])
    data, fps = load_cache(cp)
    if data is None:
        data, fps, status = analyse(path, name, k, show=True)
        if status == "quit":
            return True
        if status == "skip" or data is None:
            return False
        save_cache(cp, data, fps)
        print(f"{name}: max_prob = {data[-1][2]:.2f}   (analyse hui aur save ho gayi)" if data else f"{name}: khali")
    else:
        print(f"{name}: max_prob = {data[-1][2]:.2f}   (cache se, foran)" if data else f"{name}: khali")
        if a.save:
          try:
            export_cached(path, name, k, data, fps)
          except Exception as e:
            print(f"[error] {name}: save nahi hui -> {e}")
          return False
    return play_cached(path, name, k, data, fps)


# ---------------------------------------------------------------- live mode (purana tareeqa)
def play_live(path, name, k):
    """Har frame foran analyse + dikhao. Returns True agar 'q' dabaya."""
    global skip, proc_fps
    cap, fps = open_video(path, name)
    if cap is None:
        return False
    an = Analyzer(yolo)
    yolo.predictor = None  # tracker reset
    run = Run(fps)
    paused, quit_all, t_prev = False, False, time.time()
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            for _ in range(skip - 1):          # beech ki frames chhor do => video tez dikhti hai
                if not cap.grab():
                    break
            vec, boxes, pair = an.step(frame)
            run.feed(vec, pair, skip)
            now = time.time()
            proc_fps = 0.9 * proc_fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            cv2.imshow(WIN, annotate(frame, boxes, run, k, name,
                                     f"proc FPS {proc_fps:.1f}   imgsz {C.IMGSZ}   skip {skip}   (f = tez, s = slow)"))
        key = get_key()
        if key == ord("q"):
            quit_all = True
            break
        if key == ord("n"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("f"):
            skip += 1
            print(f"skip = {skip}")
        if key == ord("s"):
            skip = max(1, skip - 1)
            print(f"skip = {skip}")
    cap.release()
    print(f"{name}: max_prob = {run.maxp:.2f}")
    return quit_all


# ---------------------------------------------------------------- main
start = max(a.start, 1) - 1
if a.build_cache:
    for idx in range(start, len(items)):
        label, path = items[idx]
        name = os.path.join(label, os.path.basename(path)) if label else os.path.basename(path)
        cp = cache_file(label, path)
        data, _ = load_cache(cp)
        if data is not None:
            print(f"[{idx + 1}/{len(items)}] {name}: cache pehle se maujood hai")
            continue
        data, fps, status = analyse(path, name, idx + 1, show=False)
        if data:
            save_cache(cp, data, fps)
            print(f"[{idx + 1}/{len(items)}] {name}: max_prob = {data[-1][2]:.2f}  (save ho gayi)")
    raise SystemExit("Cache tayyar. Ab 'python detect_all.py --folder data --prerender' chalao -- sab foran chalegi.")

for idx in range(start, len(items)):
    label, path = items[idx]
    name = os.path.join(label, os.path.basename(path)) if label else os.path.basename(path)
    quit_all = play_prerender(path, name, idx + 1) if a.prerender else play_live(path, name, idx + 1)
    if quit_all:
        break

cv2.destroyAllWindows()
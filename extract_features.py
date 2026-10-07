"""Videos -> features.csv
Folder: data/accident/*.mp4 , data/normal/*.mp4
annotations.csv: video,start_frame,end_frame  (accident videos ke liye, 1-based frames)

Labeling (annotations ke saath):
  window ke kam az kam --min_overlap hissa frames takkar ke andar  -> label 1
  window takkar se bilkul bahar                                    -> label 0
  beech ki (thora sa overlap) windows                              -> training se nikaal di jati hain
Annotation na ho to accident video ki sari windows label=1 hongi (noisy)."""
import argparse, glob, os
import cv2, pandas as pd
import config as C
from pipeline import Analyzer, FEATURE_NAMES, load_model

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--annotations", default=None)
p.add_argument("--out", default="features.csv")
p.add_argument("--min_overlap", type=float, default=0.5,
               help="window ka kitna hissa takkar mein ho to label 1 (0-1)")
a = p.parse_args()

ann = {}
if a.annotations:
    for _, r in pd.read_csv(a.annotations).iterrows():
        ann[r["video"]] = (int(r["start_frame"]), int(r["end_frame"]))

model = load_model()
rows = []
for label_name, label in (("normal", 0), ("accident", 1)):
    paths = [f for f in sorted(glob.glob(os.path.join(a.data, label_name, "*")))
             if os.path.splitext(f)[1].lower() in VIDEO_EXT]
    for path in paths:
        name = os.path.basename(path)
        if label == 1 and a.annotations and name not in ann:
            print(f"[skip] {name}: annotations.csv mein nahi hai")
            continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[skip] {name}: video khuli nahi")
            continue
        an = Analyzer(model)
        model.predictor = None  # reset tracker per video
        idx, w, pos, skipped = 0, 0, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            vec, _ = an.step(frame)
            if vec is None:
                continue
            lo, hi = idx - C.WINDOW + 1, idx
            y = label
            if label == 1 and name in ann:
                s, e = ann[name]
                ov = max(0, min(hi, e) - max(lo, s) + 1)
                if ov == 0:
                    y = 0
                elif ov / C.WINDOW >= a.min_overlap:
                    y = 1
                else:
                    skipped += 1  # ambiguous window, training se bahar
                    continue
            pos += y
            rows.append({"video": name, "window": w, "label": y,
                         **dict(zip(FEATURE_NAMES, vec))})
            w += 1
        cap.release()
        extra = f", accident={pos}, skipped={skipped}" if label == 1 else ""
        print(f"{name}: {w} windows{extra}")

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print("saved", a.out, len(df), "windows",
      "| label1 =", int((df["label"] == 1).sum()), "| label0 =", int((df["label"] == 0).sum()))
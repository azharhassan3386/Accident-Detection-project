"""Saari videos par detection chala kar ek table banata hai (window/video nahi dikhata, tez hai).
Folder: <data>/accident/*.mp4 aur <data>/normal/*.mp4
Output: har video ki max probability, alert aaya ya nahi, aur results.csv.

Dhyan: agar data training wali videos hain to result achha dikhega (model ne unhein dekha hai).
Asli test ke liye nayi videos alag folder mein rakhein (test/accident, test/normal)
aur chalayein:  python evaluate.py --data test"""
import argparse, csv, glob, os
import cv2
from xgboost import XGBClassifier
import config as C
from pipeline import Analyzer, load_model

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--model", default="model.json")
p.add_argument("--out", default="results.csv")
a = p.parse_args()

clf = XGBClassifier()
clf.load_model(a.model)
yolo = load_model()
rows = []
print(f"ALERT_THRESHOLD={C.ALERT_THRESHOLD}  ALERT_CONSECUTIVE={C.ALERT_CONSECUTIVE}\n")
print(f"{'type':9s} {'video':34s} {'max_prob':>8s}  alert  first_alert_frame")
for label in ("accident", "normal"):
    paths = [f for f in sorted(glob.glob(os.path.join(a.data, label, "*")))
             if os.path.splitext(f)[1].lower() in VIDEO_EXT]
    for path in paths:
        name = os.path.basename(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"{label:9s} {name:34s} khuli nahi, skip")
            continue
        an = Analyzer(yolo)
        yolo.predictor = None  # tracker reset
        idx, streak, max_prob, first = 0, 0, 0.0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            vec, _ = an.step(frame)
            if vec is None:
                continue
            prob = float(clf.predict_proba(vec.reshape(1, -1))[0, 1])
            max_prob = max(max_prob, prob)
            streak = streak + 1 if prob >= C.ALERT_THRESHOLD else 0
            if streak >= C.ALERT_CONSECUTIVE and first is None:
                first = idx
        cap.release()
        alert = first is not None
        rows.append({"type": label, "video": name, "max_prob": round(max_prob, 3),
                     "alert": int(alert), "first_alert_frame": first or ""})
        print(f"{label:9s} {name:34s} {max_prob:8.2f}  {'YES' if alert else 'no ':3s}    {first or '-'}")

with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["type", "video", "max_prob", "alert", "first_alert_frame"])
    w.writeheader()
    w.writerows(rows)

acc = [r for r in rows if r["type"] == "accident"]
nor = [r for r in rows if r["type"] == "normal"]
print("\nSummary")
print(f"  Accident videos mein alert aaya : {sum(r['alert'] for r in acc)} / {len(acc)}")
print(f"  Normal videos mein ghalat alert : {sum(r['alert'] for r in nor)} / {len(nor)}")
print("saved", a.out)

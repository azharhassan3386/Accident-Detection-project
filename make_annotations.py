"""data/accident ki har video chala kar takkar ke start/end frame mark karo.
Output: annotations.csv (video,start_frame,end_frame)

Keys (video window par click kar ke dabayein):
  SPACE  play / pause
  a / d  1 frame peechay / aagay (pause mein)
  j / l  10 frames peechay / aagay
  s      is frame ko takkar ka START mark karo
  e      is frame ko takkar ka END mark karo
  n      save kar ke agli video
  k      is video ko skip karo
  q      save kar ke band
"""
import csv
import glob
import os
import cv2

FOLDER = os.path.join("data", "accident")
OUT = "annotations.csv"

done = {}
if os.path.exists(OUT):
    with open(OUT, newline="") as f:
        for r in csv.DictReader(f):
            done[r["video"]] = (int(r["start_frame"]), int(r["end_frame"]))


def save():
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "start_frame", "end_frame"])
        for v, (s, e) in done.items():
            w.writerow([v, s, e])


quit_all = False
for path in sorted(glob.glob(os.path.join(FOLDER, "*"))):
    name = os.path.basename(path)
    if name in done:
        print(f"{name}: pehle se annotated, skip")
        continue
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"{name}: khuli nahi, skip")
        continue
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pos, playing, start, end = 0, False, None, None
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            pos = max(total - 1, 0)
            playing = False
            continue
        shown = pos + 1  # extract_features.py 1-based frame numbers use karta hai
        txt = f"{name}  frame {shown}/{total}  start={start}  end={end}"
        cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Annotate", frame)
        k = cv2.waitKey(30 if playing else 0) & 0xFF
        if k == ord(" "):
            playing = not playing
        elif k == ord("a"):
            pos, playing = max(pos - 1, 0), False
        elif k == ord("d"):
            pos, playing = min(pos + 1, total - 1), False
        elif k == ord("j"):
            pos, playing = max(pos - 10, 0), False
        elif k == ord("l"):
            pos, playing = min(pos + 10, total - 1), False
        elif k == ord("s"):
            start = shown
        elif k == ord("e"):
            end = shown
        elif k == ord("n"):
            if start is None or end is None or end < start:
                print("Pehle s aur e dabayein (end, start se baad ho).")
                continue
            done[name] = (start, end)
            save()
            print(f"{name}: {start} -> {end} saved")
            break
        elif k == ord("k"):
            break
        elif k == ord("q"):
            quit_all = True
            break
        elif k == 255 and playing:  # koi key nahi, play jari rakho
            pos += 1
            if pos >= total:
                pos, playing = total - 1, False
    cap.release()
    if quit_all:
        break

cv2.destroyAllWindows()
save()
print("Done. annotations.csv mein", len(done), "videos hain.")

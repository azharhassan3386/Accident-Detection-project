# Model: YOLO26 (agar aapke ultralytics version mein nahi hai to automatic fallback hoga)
MODEL = "yolo26s.pt"        # "n" (nano) se "s" (small) -- chhoti/door gariyan behtar pakarta hai
FALLBACK_MODEL = "yolo11s.pt"

# COCO classes: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
CLASSES = [0, 1, 2, 3, 5, 7]

CONF = 0.25
IMGSZ = 1280         # 640 se badhaya -- wide/aerial traffic shots mein door gariyan chand
                      # pixels ki reh jati hain, zyada IMGSZ unhe zyada detail deta hai
WINDOW = 30          # frames per analysis window
STRIDE = 5          # new window every N frames
FLOW_SIZE = (320, 180)
OVIF_BINS = 8
ALERT_THRESHOLD = 0.4   # XGBoost probability
ALERT_CONSECUTIVE = 3   # windows in a row needed for alert
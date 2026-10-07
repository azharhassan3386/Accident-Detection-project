# YoFlow-26: Scenario-Based Automatic Traffic Accident Detection
YOLO26 + ByteTrack + Dense Optical Flow (Farneback) + OViF + XGBoost

## Setup
    pip install -r requirements.txt

## Data layout
    data/accident/*.mp4
    data/normal/*.mp4
    annotations.csv   (optional) video,start_frame,end_frame

Public datasets: CADP, DoTA, TAD, UCF-Crime (road accident subset).

## Run
    python extract_features.py --data data --annotations annotations.csv
    python train.py
    python detect.py --source test.mp4        # ya --source 0 webcam

## Notes
- Train/test split video-wise hai (GroupKFold), taake naye videos par honest score mile.
- Thresholds config.py mein hain. Results aapke data par depend karenge; pehle se koi accuracy claim nahi.
- Scenario-based evaluation: apne test videos ko rear-end / side / pedestrian / night / rain mein tag kar ke alag-alag precision/recall/F1 nikalein.

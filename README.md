# Accident Detection from Traffic Video

Real-time traffic accident detection from regular CCTV footage using
YOLO tracking, optical flow, and an XGBoost classifier.

![demo](demo.gif)

## How it works

1. **Detection and tracking:** YOLO detects and tracks vehicles frame by frame.
2. **Motion features:** Optical flow (OViF histograms over a sliding window)
   captures sudden changes in motion between vehicles.
3. **Classification:** An XGBoost model (`model.json`) scores each window
   as accident or normal.
4. **Alert:** If the probability stays above the threshold for several
   consecutive windows, an alert is raised and the colliding pair is
   marked with a red box.

## Project structure

| File | Purpose |
|------|---------|
| `config.py` | Thresholds and settings (confidence, image size, window, alert threshold) |
| `pipeline.py` | Main pipeline: tracking, optical flow, features, prediction |
| `detect.py` | Run detection on a single video |
| `detect_all.py` | Play and analyze all videos in a folder |
| `extract_features.py` | Build the feature dataset from videos |
| `make_annotations.py` | Create annotations for training |
| `evaluate.py` | Evaluate the model |
| `model.json` | Trained XGBoost model |

## Installation

```bash
git clone https://github.com/azharhassan3386/Accident-Detection-project.git
cd Accident-Detection-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run on a single video:
```bash
python detect.py --video data/a.mp4
```

Run on a whole folder:
```bash
python detect_all.py --folder data
```
Keys: `n` next video, `SPACE` pause, `q` quit.

## Results

- Tested on: [N] clips ([9] accident, [6] normal)
- Accuracy / Precision / Recall: [fill from evaluate.py]
- Processing speed: [X] FPS on [CPU/GPU]

## Limitations

- Processing is slow on CPU at high image size.
- Heavy congestion or sudden braking can cause false alarms.
- Trained on a small dataset, so results may vary on new camera angles.

## Future work

- Larger and more varied dataset
- Faster inference (GPU, smaller image size)
- Real-time RTSP camera input

## Author

Azhar Hassan, [www.linkedin.com/in/azhar-hassan-581b253ba]

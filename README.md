# Staff Detection System

Automatically detect and count **staff members** in video footage using a two-stage deep-learning pipeline:

1. **Person Detection** — YOLOv11 locates every person in each frame
2. **Staff Classification** — Swin Transformer classifies each crop as staff or non-staff

---

## How It Works

```
Video Frame → YOLOv11 (person detection) → Crop each person
           → Swin Transformer (staff/non-staff) → Annotated output video
```

Staff are identified by their uniform / nametag appearance. Crops classified above the confidence threshold (`--thres`) are labelled as staff and highlighted in the output video.

---

## Project Structure

```
staff_detect/
├── detect.py               # Main inference script (staff detection)
├── train.py                # Person tracking & counting (no classifier)
├── employee_detection.py   # Filters YOLO output → bounding boxes
├── id_classification.py    # Runs Swin Transformer classifier on crops
├── earlystopping.py        # Early stopping helper (training utility)
├── self_transformers.py    # Swin Transformer model definition
├── requirements.txt        # CPU dependencies
├── requirements_gpu.txt    # GPU dependencies (CUDA 11.8 + PyTorch 2.3)
├── yolo11m.pt              # YOLOv11 medium weights (person detector)
├── staff_class.pt          # Staff classifier checkpoint (Swin Transformer)
└── sample.mp4              # Example input video
```

Output is written to `runs/detect/runN/` (auto-incremented each run).

---

## Installation

### CPU

```bash
pip install -r requirements.txt
```

### GPU (CUDA 11.8 + PyTorch 2.3)

```bash
pip install -r requirements_gpu.txt
pip install -r requirements.txt
```

> **Python 3.8–3.11** recommended.

---

## Usage

### Staff Detection (`detect.py`)

Detects and counts staff in a video. Saves an annotated output video and cropped detections by default.

```bash
python detect.py \
    --detection_model yolo11m.pt \
    --staff_model     staff_class.pt \
    --video           sample.mp4 \
    --thres           0.3 \
    --det_thres       0.4
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--detection_model` | `yolo11m.pt` | YOLOv11 weights file |
| `--staff_model` | *(required)* | Path to staff classifier checkpoint |
| `--video` | `sample.mp4` | Input video path |
| `--thres` | `0.7` | Staff score threshold (lower = more detections) |
| `--det_thres` | `0.8` | Person detection confidence threshold |
| `--no_crop` | off | Add flag to disable saving cropped detections |
| `--view` | off | Add flag to show live preview window |

Recommended command for balanced accuracy/speed:

```bash
python detect.py --detection_model yolo11m.pt --staff_model staff_class.pt \
                 --video sample.mp4 --thres 0.3 --det_thres 0.4 --view
```

---

### Person Tracking Only (`train.py`)

Runs person detection and centroid tracking without staff classification. Useful for counting and verifying raw detections.

```bash
python train.py \
    --weights yolo11m.pt \
    --source  sample.mp4 \
    --view-img
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--weights` | `yolo11m.pt` | YOLOv11 weights |
| `--source` | `sample.mp4` | Video path, `0` for webcam, or RTSP URL |
| `--conf-thres` | `0.25` | Detection confidence threshold |
| `--iou-thres` | `0.45` | NMS IoU threshold |
| `--img-size` | `640` | Inference image size |
| `--view-img` | off | Show live preview |
| `--nosave` | off | Skip saving output video |
| `--device` | auto | `cpu`, `0` (GPU 0), `0,1` (multi-GPU) |

---

## Output

Each run creates a folder at `runs/detect/runN/`:

```
runs/detect/run1/
├── result.avi        # Annotated output video
└── crops/            # Cropped staff detections (one image per detection)
```

---

## Models

| Model | File | Purpose |
|---|---|---|
| YOLOv11 medium | `yolo11m.pt` | Detects all persons (COCO class 0) |
| Swin Transformer Base | `staff_class.pt` | Binary staff / non-staff classifier |

> **Important:** The staff classifier (`staff_class.pt`) was trained on **raw [0, 1] pixel values** — do not apply ImageNet normalisation when preprocessing crops.

---

## Tips

- **Too many false positives** → raise `--thres` (e.g. `0.5` or `0.7`)
- **Missing staff** → lower `--thres` (e.g. `0.2`) or lower `--det_thres`
- **Slow processing** → add `--skip_frames 2` to process every other frame
- Crops are saved by default — add `--no_crop` to disable if disk space is a concern

---

## Requirements

- Python 3.8–3.11
- PyTorch ≥ 2.0
- ultralytics ≥ 8.3.0
- OpenCV, Pillow, NumPy (see `requirements.txt`)

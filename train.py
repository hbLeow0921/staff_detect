"""
Person Detection & Counting with Centroid Tracking — YOLOv11 Edition.

Runs YOLOv11 person detection on a video (or webcam / RTSP stream),
tracks individuals across frames using a centroid tracker, and counts
both the current and total unique persons seen.

Usage
-----
    python train2.py \
        --weights yolo11m.pt \
        --source sample.mp4 \
        --view-img

    # Webcam
    python train2.py --weights yolo11m.pt --source 0 --view-img
"""

import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Centroid Tracker  (unchanged — pure NumPy, no YOLOv7 dependency)
# ---------------------------------------------------------------------------
class CentroidTracker:
    """Simple centroid-based multi-object tracker."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 50):
        self.next_object_id = 0
        self.objects = OrderedDict()       # id -> centroid
        self.boxes = OrderedDict()         # id -> (x1,y1,x2,y2)
        self.disappeared = OrderedDict()   # id -> missed-frame count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box):
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """Update tracker with a list of (x1, y1, x2, y2) bounding boxes."""
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects, self.boxes

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i])
            return self.objects, self.boxes

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis]
            - input_centroids[np.newaxis, :],
            axis=2,
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.boxes[oid] = rects[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(D.shape[0])).difference(used_rows):
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in set(range(D.shape[1])).difference(used_cols):
            self.register(input_centroids[col], rects[col])

        return self.objects, self.boxes


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_box(img, x1, y1, x2, y2, label, color=(0, 255, 0), thickness=2):
    """Draw a bounding box with a label above it."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 5), font, scale,
                (0, 0, 0), thick, cv2.LINE_AA)


def overlay_counts(img, current, total):
    """Draw current / total person counts on the frame."""
    cv2.putText(img, f"Current: {current}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Total Unique: {total}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Auto-increment run directory
# ---------------------------------------------------------------------------
def make_run_folder(project: str, name: str) -> str:
    """Create project/name, project/name2, … as needed."""
    base = Path(project) / name
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    idx = 2
    while True:
        candidate = Path(project) / f"{name}{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)
        idx += 1


# ---------------------------------------------------------------------------
# Main detection + tracking loop
# ---------------------------------------------------------------------------
def detect(opt):
    source = opt.source
    save_dir = make_run_folder(opt.project, opt.name)
    labels_dir = os.path.join(save_dir, "labels")
    if opt.save_txt:
        os.makedirs(labels_dir, exist_ok=True)

    # --- Determine source type ---------------------------------------------
    is_webcam = (
        source.isnumeric()
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    # --- Device selection --------------------------------------------------
    device = opt.device if opt.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load YOLOv11 model ------------------------------------------------
    print(f"Loading YOLOv11 model: {opt.weights}")
    model = YOLO(opt.weights)

    # --- Open video / webcam -----------------------------------------------
    cap_source = int(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {source}  |  {width}x{height} @ {fps:.1f} fps  |  {total_frames} frames")

    # --- Video writer ------------------------------------------------------
    vid_writer = None
    save_path = None
    if not opt.nosave:
        stem = Path(source).stem if not is_webcam else "stream"
        save_path = os.path.join(save_dir, f"{stem}.mp4")
        vid_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    # --- Tracker -----------------------------------------------------------
    tracker = CentroidTracker(max_disappeared=40, max_distance=60)
    counted_ids = set()

    # --- Processing loop ---------------------------------------------------
    frame_idx = 0
    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLOv11 inference
        results = model.predict(
            source=frame_rgb,
            imgsz=opt.img_size,
            conf=opt.conf_thres,
            iou=opt.iou_thres,
            classes=[0],          # person only
            verbose=False,
            device=device,
        )
        det = results[0]

        # Collect person bounding boxes
        person_rects = []
        boxes = det.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                person_rects.append((x1, y1, x2, y2))

                # Save label text
                if opt.save_txt:
                    # Normalised xywh format
                    cx = ((x1 + x2) / 2) / width
                    cy = ((y1 + y2) / 2) / height
                    bw = (x2 - x1) / width
                    bh = (y2 - y1) / height
                    line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                    if opt.save_conf:
                        line += f" {conf:.4f}"
                    txt_path = os.path.join(labels_dir, f"{Path(source).stem}_{frame_idx}.txt")
                    with open(txt_path, "a") as f:
                        f.write(line + "\n")

        # Update tracker
        objects, tracked_boxes = tracker.update(person_rects)

        # Draw tracked persons
        for oid, (bx1, by1, bx2, by2) in tracked_boxes.items():
            counted_ids.add(oid)
            draw_box(frame, bx1, by1, bx2, by2, f"Person {oid}", color=(0, 255, 0))

        # Overlay counts
        overlay_counts(frame, len(tracked_boxes), len(counted_ids))

        # Print per-frame summary
        n_det = len(person_rects)
        print(f"Frame {frame_idx}: {n_det} person(s) detected, "
              f"{len(tracked_boxes)} tracked, {len(counted_ids)} unique total")

        # Show live preview
        if opt.view_img:
            cv2.imshow("YOLOv11 Person Tracking", frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

        # Write to output video
        if vid_writer is not None:
            vid_writer.write(frame)

    # --- Cleanup -----------------------------------------------------------
    elapsed = time.time() - t_start
    cap.release()
    if vid_writer is not None:
        vid_writer.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} FPS)")
    print(f"Total unique persons: {len(counted_ids)}")
    if save_path:
        print(f"Output saved to: {save_path}")
    print(f"Run folder: {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Person detection & counting with YOLOv11 + centroid tracking")

    parser.add_argument("--weights", type=str, default="yolo11m.pt",
                        help="YOLOv11 model path (default: yolo11m.pt)")
    parser.add_argument("--source", type=str, default="sample.mp4",
                        help="Video file, webcam index (0), or stream URL")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Inference image size in pixels (default: 640)")
    parser.add_argument("--conf-thres", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou-thres", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: cpu, 0, cuda:0, etc. (auto if blank)")
    parser.add_argument("--view-img", action="store_true",
                        help="Show live preview window")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save detection labels to .txt files")
    parser.add_argument("--save-conf", action="store_true",
                        help="Include confidence in saved labels")
    parser.add_argument("--nosave", action="store_true",
                        help="Do not save output video")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Output project directory (default: runs/detect)")
    parser.add_argument("--name", type=str, default="exp",
                        help="Run name inside project (default: exp)")

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt)

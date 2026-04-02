import argparse
import os
import re
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

from id_classification import ID_Classificaiton
from employee_detection import EE_Detection


def error(img1, img2):
    h_arr = []
    w_arr = []

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h_arr.append(img1.shape[0])
    w_arr.append(img1.shape[1])

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h_arr.append(img2.shape[0])
    w_arr.append(img2.shape[1])

    h = max(h_arr)
    w = max(w_arr)

    img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / float(h * w)
    return mse


def make_run_folder(base_dir="runs/detect"):
    os.makedirs(base_dir, exist_ok=True)
    folder_arr = [f for f in os.listdir(base_dir) if re.match(r"^run\d+$", f)]

    if len(folder_arr) == 0:
        folder_name = os.path.join(base_dir, "run1")
    else:
        last_run = max(int(re.search(r"\d+", f).group()) for f in folder_arr)
        folder_name = os.path.join(base_dir, f"run{last_run + 1}")

    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(os.path.join(folder_name, "crops"), exist_ok=True)
    return folder_name


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def draw_label(img, text, x1, y1, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_y1 = max(0, y1 - th - baseline - 8)
    box_y2 = max(th + baseline + 8, y1)
    box_x2 = x1 + tw + 10

    cv2.rectangle(img, (x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 5, box_y2 - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_model", type=str, default="yolo11m.pt", help="YOLO model path")
    parser.add_argument("--video", type=str, default="sample.mp4", help="video path")
    parser.add_argument("--thres", type=float, default=0.7, help="confidence score for staff classification")
    parser.add_argument("--det_thres", type=float, default=0.8, help="confidence score for employee detection")
    parser.add_argument("--staff_model", type=str, required=True, help="staff model path/name")
    parser.add_argument("--no_crop", action="store_true", help="disable saving cropped staff detections")
    parser.add_argument("--view", action="store_true", help="show live preview")
    opt = parser.parse_args()

    folder_name = make_run_folder("runs/detect")
    crops_dir = os.path.join(folder_name, "crops")

    # Initialize models
    staff_model = ID_Classificaiton(opt.staff_model)
    detection_model = YOLO(opt.detection_model)

    ee_detector = EE_Detection(thres=opt.det_thres)

    cap = cv2.VideoCapture(opt.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {opt.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(folder_name, f"{Path(opt.video).stem}_output.mp4")
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0
    total_staff_detections = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # YOLOv11 inference
        results = detection_model.predict(
            source=frame_rgb,
            conf=opt.det_thres,
            classes=[0],
            verbose=False,
        )
        det_result = results[0]

        coords = ee_detector.detect(det_result, frame_rgb.shape)

        frame_staff_count = 0

        if coords:
            for det_id, pts in enumerate(coords):
                if len(pts) < 4:
                    continue

                x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop_rgb = frame_rgb[y1:y2, x1:x2]
                if crop_rgb.size == 0:
                    continue

                score = float(staff_model.output(crop_rgb))

                if score > opt.thres:
                    frame_staff_count += 1
                    total_staff_detections += 1

                    label = f"Staff {frame_staff_count} | {score:.2f}"

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    draw_label(frame_bgr, label, x1, y1, color=(0, 255, 0))

                    if not opt.no_crop:
                        crop_name = f"{frame_count}_{det_id}_{x1}_{y1}_{x2}_{y2}.jpg"
                        Image.fromarray(crop_rgb).save(os.path.join(crops_dir, crop_name))

                    print(
                        f"Staff detected on frame {frame_count} "
                        f"located at (x1,y1,x2,y2): {[x1, y1, x2, y2]} "
                        f"score={score:.4f}"
                    )
                else:
                    label = f"Non-staff | {score:.2f}"
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    draw_label(frame_bgr, label, x1, y1, color=(0, 0, 255))

        cv2.putText(frame_bgr, f"Frame: {frame_count}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Staff in frame: {frame_staff_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Total staff detections: {total_staff_detections}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame_bgr)

        if opt.view:
            cv2.imshow("Detection", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\nSaved output video to: {output_path}")
    print(f"Saved run folder: {folder_name}")


if __name__ == "__main__":
    main()

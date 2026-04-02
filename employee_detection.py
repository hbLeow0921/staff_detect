"""
Employee Detection Module — YOLOv11 (ultralytics) compatible.

Filters YOLO detection results for person-class bounding boxes
that exceed a confidence threshold.
"""

from typing import List, Union


class EE_Detection:
    """Extract person bounding boxes from YOLOv11 results."""

    PERSON_CLASS_ID = 0  # COCO class index for 'person'

    def __init__(self, thres: float = 0.8):
        self.threshold = thres

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def detect(self, results, frame_shape) -> Union[List[List[int]], bool]:
        """
        Parameters
        ----------
        results : ultralytics.engine.results.Results
            A single Results object returned by model(frame)[0].
        frame_shape : tuple
            (height, width, channels) of the source frame.

        Returns
        -------
        list[list[int]]  – [[x1,y1,x2,y2], …] for every person detection, or
        False            – when no persons are found.
        """
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return False

        h, w = frame_shape[:2]
        coords = []

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if cls_id != self.PERSON_CLASS_ID:
                continue
            if conf < self.threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Clamp to frame boundaries
            x1 = max(1, min(int(x1), w))
            y1 = max(1, min(int(y1), h))
            x2 = max(1, min(int(x2), w))
            y2 = max(1, min(int(y2), h))

            if x2 > x1 and y2 > y1:
                coords.append([x1, y1, x2, y2])

        return coords if coords else False

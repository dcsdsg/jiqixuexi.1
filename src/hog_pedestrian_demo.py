# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


def real_time_pedestrian_masking(mask_path: Path | None = None, camera_index: int = 0) -> None:
    import cv2

    base_dir = Path(__file__).resolve().parents[1]
    mask_path = mask_path or (base_dir / "mask.png")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    mask_img = cv2.imread(str(mask_path))
    if mask_img is None:
        print(f"Warning: mask image not found at {mask_path}, using black boxes instead.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Unable to open camera")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(32, 32), scale=1.05)

        for (x, y, w, h) in boxes:
            if mask_img is not None:
                resized_mask = cv2.resize(mask_img, (w, h))
                frame[y:y + h, x:x + w] = resized_mask
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

        cv2.imshow("Pedestrian Masking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    real_time_pedestrian_masking()


if __name__ == "__main__":
    main()
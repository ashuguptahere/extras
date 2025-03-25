import cv2
import torch
import numpy as np
from ultralytics import YOLO, SAM
from segment_anything import SamPredictor, sam_model_registry

# Load YOLOv8 model (Ensure you have the right model weights)
yolo_model = YOLO("yolo11x.pt")

# Load SAM model (Download the correct model checkpoint for SAM v2.1)
sam_checkpoint = "sam_vit_b_01ec64.pth"  # Replace with correct path
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)  # Use ViT-H model
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
sam_predictor = SamPredictor(sam)

# Open video capture
cap = cv2.VideoCapture("inputs/input.mp4")  # 0 for webcam, or provide video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # Extract detections

    # Filter for 'person' class (COCO class 0 for person)
    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) == 0 and score > 0.3:  # Only process 'person'
            # Extract person region
            person_crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
            if person_crop.size == 0:
                continue

            # Run SAM segmentation
            sam_predictor.set_image(frame)
            masks, _, _ = sam_predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                point_coords=None,
                point_labels=None,
                multimask_output=False,
            )

            # Overlay mask on the frame
            for mask in masks:
                mask = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, (0, 255, 0), thickness=cv2.FILLED)

    # Display the frame
    cv2.imshow("Person Detection with SAM", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

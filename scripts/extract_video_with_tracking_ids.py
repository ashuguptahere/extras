import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Initialize YOLO with the yolo11x model
model = YOLO("yolo11x.pt")

# Input video path
video_path = "videos/FaceGuard take-1.mp4"
cap = cv2.VideoCapture(video_path)

# Output video dimensions and FPS
output_width, output_height = 384, 640
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Store video writers for each person
person_videos = {}
track_history = defaultdict(lambda: [])

# Output directory
os.makedirs("outputs", exist_ok=True)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, classes=[0])

    # Get the bounding boxes and track IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Process each person detected in the frame
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)  # bounding box coordinates

        # Crop the person from the frame
        person_frame = frame[y1:y2, x1:x2]

        # Calculate the aspect-ratio-preserving resize dimensions
        h, w = person_frame.shape[:2]
        scale = min(output_width / w, output_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_person_frame = cv2.resize(person_frame, (new_w, new_h))

        # Create a blank frame to paste the resized person frame onto
        output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        x_offset = (output_width - new_w) // 2
        y_offset = (output_height - new_h) // 2
        output_frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            resized_person_frame
        )

        # Initialize VideoWriter for each new track_id
        if track_id not in person_videos:
            person_video_path = f"outputs/person_{track_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            person_videos[track_id] = cv2.VideoWriter(
                person_video_path, fourcc, fps, (output_width, output_height)
            )

        # Write the frame to the corresponding person video
        person_videos[track_id].write(output_frame)

# Release resources
cap.release()
for writer in person_videos.values():
    writer.release()
cv2.destroyAllWindows()

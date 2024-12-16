import os
import cv2
from ultralytics import RTDETR

# Path configurations
images_path = "images"  # Folder to save frames
labels_path = "labels"  # Folder to save annotations
videos_path = "videos"  # Folder containing input videos

# Ensure directories exist
os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)

# Load RTDETR model (ensure you have the correct weights for detecting people)
model = RTDETR("rtdetr-x.pt")

# Process each video in the folder
for video_file in os.listdir(videos_path):
    video_path = os.path.join(videos_path, video_file)
    if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 1 second frame
        if frame_count % fps == 0:
            frame_name = (
                f"{os.path.splitext(video_file)[0]}_frame_{frame_count // fps:05}.jpg"
            )
            frame_path = os.path.join(images_path, frame_name)

            # Perform object detection
            results = model.track(frame, conf=0.4)  # Adjust confidence threshold if needed

            # Save the frame
            cv2.imwrite(frame_path, frame)

            # Save annotations
            label_path = os.path.join(
                labels_path, f"{os.path.splitext(frame_name)[0]}.txt"
            )
            with open(label_path, "w") as label_file:
                for result in results[0].boxes:
                    cls, conf, xyxy = result.cls, result.conf, result.xyxy
                    if cls == 0:  # Class 0 corresponds to 'person' in COCO dataset
                        x_min, y_min, x_max, y_max = map(int, xyxy[0].tolist())
                        label_file.write(
                            f"{int(cls)} {float(conf):.2f} {x_min} {y_min} {x_max} {y_max}\n"
                        )

        frame_count += 1

    cap.release()

print("Processing complete. Frames and labels saved.")

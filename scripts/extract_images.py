import os
import cv2
from tqdm import tqdm
from ultralytics import RTDETR

# Path configurations
images_path = "images2"  # Folder to save frames
labels_path = "labels2"  # Folder to save annotations
videos_path = "videos2"  # Folder containing input videos

# Ensure directories exist
os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)

# Load RTDETR model (ensure you have the correct weights for detecting people)
model = RTDETR("rtdetr-x.pt")

# Process each video in the folder
for video_file in tqdm(os.listdir(videos_path)):
    video_path = os.path.join(videos_path, video_file)

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
        if frame_count % (fps) == 0:
            frame_name = (
                f"{os.path.splitext(video_file)[0]}_frame_{frame_count // fps:05}.jpg"
            )
            frame_path = os.path.join(images_path, frame_name)

            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()

print("Processing complete. Frames and labels saved.")

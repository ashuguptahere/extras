import os
import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO
from collections import defaultdict

# Initialize YOLO with the yolo11x model
model = YOLO("yolo11x.pt")

# Input video path
# video_path = "inputs/input.mp4"
files = glob("datasets/KTH_Action_Dataset/*/*")

for file in files:
    cap = cv2.VideoCapture(file)

    # Output video dimensions and FPS
    output_width, output_height = 384, 640
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Store video writers and frame counts for each person
    person_videos = {}
    frame_counts = defaultdict(int)
    video_counters = defaultdict(int)

    # Output directory
    os.makedirs("outputs", exist_ok=True)

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0])

        # Safeguard: Check if there are valid detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Attempt to get the bounding boxes and track IDs
            try:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
            except AttributeError:
                # If any attributes are missing, skip processing this frame
                continue

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
                output_frame = np.zeros(
                    (output_height, output_width, 3), dtype=np.uint8
                )
                x_offset = (output_width - new_w) // 2
                y_offset = (output_height - new_h) // 2
                output_frame[
                    y_offset : y_offset + new_h, x_offset : x_offset + new_w
                ] = resized_person_frame

                # Check if a new video is needed for the track_id
                if frame_counts[track_id] % 30 == 0:
                    # Close the existing video writer if it exists
                    if track_id in person_videos and person_videos[track_id]:
                        person_videos[track_id].release()

                    # Increment video counter and initialize a new video writer
                    video_counters[track_id] += 1
                    person_video_path = (
                        f"outputs/person_{track_id}_part_{video_counters[track_id]}.mp4"
                    )
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    person_videos[track_id] = cv2.VideoWriter(
                        person_video_path, fourcc, fps, (output_width, output_height)
                    )

                # Write the frame to the current person's video
                person_videos[track_id].write(output_frame)
                frame_counts[track_id] += 1

    # Release resources
    cap.release()
    for writer in person_videos.values():
        if writer:
            writer.release()
    cv2.destroyAllWindows()

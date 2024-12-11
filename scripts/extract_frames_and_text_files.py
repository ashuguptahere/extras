import os
import cv2
from ultralytics import YOLO

# Load the YOLO model or another model, depending on your preference
model = YOLO("yolo11x.pt")


def process_video(video_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # # Save the current frame
        # frame_name = f"frame_{frame_idx}.jpg"
        # frame_path = os.path.join(output_dir, frame_name)
        # cv2.imwrite(frame_path, frame)

        # Detect persons on the frame
        results = model(frame)

        # Filter detections for 'person' class and save in a text file
        txt_name = f"frame_{frame_idx}.txt"
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, "w") as f:
            for result in results:
                for box in result.boxes:
                    # If the detected object is a person, save its bounding box
                    if box.cls == 0:  # Assuming 'person' is class 0
                        x_min, y_min, x_max, y_max = box.xyxy[0]
                        f.write(
                            f"{x_min.item()} {y_min.item()} {x_max.item()} {y_max.item()}\n"
                        )

        frame_idx += 1

    # Release the video capture object
    cap.release()


# Usage
process_video("inputs/input.mp4", "text_files")

import os
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11x.pt")

# Input and output paths
input_path = "videos/input.mp4"
output_path = "outputs/detected_person.mp4"

# Output video dimensions (384x640)
output_width, output_height = 384, 640

# Create the output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Open the input video
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model inference on the frame
    results = model(frame)

    # Filter detections for the 'person' class
    for result in results[0].boxes:
        if int(result.cls) == 0:  # 'person' class in COCO dataset is represented by 0
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            # Crop the detected person region
            person_frame = frame[y1:y2, x1:x2]

            # Calculate aspect-ratio-preserving size
            h, w = person_frame.shape[:2]
            scale = min(output_width / w, output_height / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize the person frame
            resized_person_frame = cv2.resize(person_frame, (new_w, new_h))

            # Create a blank frame with the output size
            output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

            # Calculate position to center the resized person frame
            x_offset = (output_width - new_w) // 2
            y_offset = (output_height - new_h) // 2

            # Place the resized person frame on the blank frame
            output_frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
                resized_person_frame
            )

            # Write the frame to the output video
            out.write(output_frame)

    # Optional: Display the frame (comment out in production)
    # cv2.imshow("Detected Person", output_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

from ultralytics import RTDETR
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load pre-trained YOLO model for detection
model = RTDETR("rtdetr-x.pt")

# Initialize DeepSORT
deep_sort = DeepSort()

image_paths = "../../Combined2/images"
labels = "../../Combined2/labels"

# Loop through the images and labels (assuming you have list of bounding boxes)
for image_path, label in zip(image_paths, labels):
    # Perform detection if necessary
    detections = model(image_path)  # In case you want to re-detect objects

    # Pass bounding boxes to DeepSORT (format: [x1, y1, x2, y2])
    tracks = deep_sort.update_tracks(detections)

    # Get unique tracking IDs
    for track in tracks:
        track_id = track[1]  # the tracking ID for each object
        print(f"Person ID: {track_id}")

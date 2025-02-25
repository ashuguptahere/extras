import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Define paths
IMAGES_DIR = "images"
LABELS_DIR = "labels"
NEW_LABELS_DIR = "labels_new"
MODEL_PATH = "../shoplifting-v2/runs/detect/train3/weights/best.pt"

# Create output labels directory if it doesn't exist
os.makedirs(NEW_LABELS_DIR, exist_ok=True)

# Load a model
model = YOLO(MODEL_PATH)

# Process each image
for img_file in tqdm(sorted(os.listdir(IMAGES_DIR))):
    image_path = os.path.join(IMAGES_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + ".txt")
    new_label_path = os.path.join(
        NEW_LABELS_DIR, os.path.splitext(img_file)[0] + ".txt"
    )

    # Skip if label file does not exist
    if not os.path.exists(label_path):
        continue

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        continue

    # Read old label file
    with open(label_path, "r") as f:
        old_labels = [line.strip().split() for line in f.readlines()]

    # Get bounding boxes in (x_center, y_center, width, height) format
    bboxes = [[float(x) for x in label[1:]] for label in old_labels]

    # Run inference on image
    results = model(image)[0]

    # Assign new class labels based on highest confidence detection per bounding box
    new_labels = []
    for i, bbox in enumerate(bboxes):
        x_center, y_center, width, height = bbox

        # Find the most confident class prediction
        if len(results.boxes) > i:
            new_class = int(results.boxes[i].cls.item())  # Get class prediction
        else:
            new_class = int(old_labels[i][0])  # Fallback to old class if no detection

        # Keep same bounding box but replace the class
        new_labels.append(f"{new_class} {x_center} {y_center} {width} {height}")

    # Write new labels to file
    with open(new_label_path, "w") as f:
        f.write("\n".join(new_labels))

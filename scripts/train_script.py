import os
from ultralytics import YOLO

# Get current working directory
cwd = os.getcwd()

# Load a COCO-pretrained YOLO model
model = YOLO("rtdetr-x.yaml")

# Train the model
results = model.train(
    data="/home/qulith-jr/Desktop/QL/datasets/dataset-normal/data.yaml",
    epochs=500,
    imgsz=640,
    device=1,
    batch=0.95,
)
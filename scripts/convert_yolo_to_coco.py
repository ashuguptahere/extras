import os
import json
from glob import glob
from PIL import Image


# Define the COCO format structure
def create_coco_structure():
    return {
        "info": {
            "description": "YOLO to COCO Conversion",
            "version": "1.0",
            "year": 2025,
            "contributor": "User",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }


def yolo_to_coco(yolo_folder, image_folder, output_json, category_names):
    coco = create_coco_structure()

    # Add category information
    for idx, name in enumerate(category_names):
        coco["categories"].append(
            {"id": idx + 1, "name": name, "supercategory": "none"}
        )

    annotation_id = 1
    image_id = 1

    # Supported image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    for txt_file in glob(os.path.join(yolo_folder, "*.txt")):
        base_name = os.path.basename(txt_file).replace(".txt", "")

        # Find the corresponding image file
        image_path = None
        for ext in image_extensions:
            potential_path = os.path.join(image_folder, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if not image_path:
            continue  # Skip if corresponding image is not found

        image_filename = os.path.basename(image_path)

        # Get actual image dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height,
            }
        )

        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                # Convert to absolute COCO format
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                bbox_width *= width
                bbox_height *= height

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,  # COCO category IDs start from 1
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        image_id += 1

    with open(output_json, "w") as json_file:
        json.dump(coco, json_file, indent=4)

    print(f"COCO JSON saved to {output_json}")


# Process train, test, and val folders
data_splits = [
    "/home/qulith-jr/Desktop/QL/datasets/dataset/train",
    "/home/qulith-jr/Desktop/QL/datasets/dataset/test",
    "/home/qulith-jr/Desktop/QL/datasets/dataset/val",
]
for split in data_splits:
    yolo_to_coco(
        yolo_folder=f"{split}/labels",  # Folder containing YOLO .txt label files
        image_folder=f"{split}/images",  # Folder containing corresponding images
        output_json=f"{split}/_annotations.coco.json",
        category_names=[
            "carry/hold",
            "crouch/kneel",
            "grab",
            "shoplift",
            "stand",
            "talk",
            "walk",
            "violence",
            "weapon",
        ],  # Updated class names
    )

# Updating normalized yolo format to yolo format
# from
# 0 0.3 0.5 0.3 0.4
# to
# 0 220 220 440 440

import os

# Input and output directories
images_dir = "/home/qulith-jr/Downloads/gun-object-detection/Images"
labels_dir = "/home/qulith-jr/Downloads/gun-object-detection/Labels"
output_labels_dir = "/home/qulith-jr/Downloads/gun-object-detection/YOLO_Labels"

# Create output directory if not exists
os.makedirs(output_labels_dir, exist_ok=True)

# Process each label file
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, label_file)
    output_label_path = os.path.join(output_labels_dir, label_file)

    with open(label_path, "r") as f:
        lines = f.readlines()

    num_objects = int(lines[0].strip())  # First line is the number of objects
    image_name = label_file.replace(".txt", ".jpeg")
    image_path = os.path.join(images_dir, image_name)

    # Get image dimensions
    from PIL import Image

    image = Image.open(image_path)
    img_width, img_height = image.size

    yolo_labels = []

    for line in lines[1 : num_objects + 1]:
        x1, y1, x2, y2 = map(int, line.strip().split())

        # Convert to YOLO format
        x_center = (x1 + x2) / (2 * img_width)
        y_center = (y1 + y2) / (2 * img_height)
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write YOLO format labels
    with open(output_label_path, "w") as f:
        f.write("\n".join(yolo_labels))

print("Conversion completed! YOLO labels saved in 'YOLO_Labels' folder.")

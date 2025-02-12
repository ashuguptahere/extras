import os
import cv2

# Paths to directories
images_path = "images"  # Folder containing images
labels_path = "labels"  # Folder containing labels

# Ensure the directories exist
if not os.path.exists(images_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("Ensure 'images' and 'labels' directories exist.")

# Get sorted list of image files
image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

if not image_files:
    raise FileNotFoundError("No images found in the 'images' directory.")

# Start with the first image
current_index = 0

while True:
    # Get the current image and label file
    image_file = image_files[current_index]
    image_path = os.path.join(images_path, image_file)
    label_file = os.path.join(labels_path, os.path.splitext(image_file)[0] + ".txt")

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_file}, skipping.")
        continue

    # Read the label file and plot bounding boxes
    if os.path.exists(label_file):
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls, conf, x_min, y_min, x_max, y_max = map(float, parts)

                # Draw bounding box and label on the image
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, f"{int(cls)} ({conf:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the image
    cv2.imshow('Image with Bounding Boxes', image)

    # Wait for a key press
    key = cv2.waitKey(0)

    if key == 27 or key == ord('q'):  # Press 'Esc' to exit
        break
    elif key == ord('n'):  # Press 'n' for next
        current_index = (current_index + 1) % len(image_files)
    elif key == ord('p'):  # Press 'p' for previous
        current_index = (current_index - 1 + len(image_files)) % len(image_files)

cv2.destroyAllWindows()

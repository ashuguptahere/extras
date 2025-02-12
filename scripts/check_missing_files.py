import os
import shutil

# Define the folder paths
base_dir = "../../BATCH_08_01_2025"
images_folder = os.path.join(base_dir, "images")
labels_folder = os.path.join(base_dir, "labels")
missing_images_folder = os.path.join(base_dir, "missing_images")
missing_labels_folder = os.path.join(base_dir, "missing_labels")

# Create the missing directories if they don't exist
os.makedirs(missing_images_folder, exist_ok=True)
os.makedirs(missing_labels_folder, exist_ok=True)

# List all files in both folders
images_files = set(os.listdir(images_folder))
labels_files = set(os.listdir(labels_folder))

# Extract the file names without extensions
images_base_names = {os.path.splitext(f)[0] for f in images_files}
labels_base_names = {os.path.splitext(f)[0] for f in labels_files}
print(len(images_base_names))
print(len(labels_base_names))

# Find missing files
missing_in_images = labels_base_names - images_base_names
missing_in_labels = images_base_names - labels_base_names

# Display the results
print("Missing in images:", len(missing_in_images))
print("Missing in labels:", len(missing_in_labels))

# Move missing images
for base_name in missing_in_images:
    label_file = f"{base_name}.txt"  # Assuming label files have .txt extension
    if label_file in labels_files:
        label_path = os.path.join(labels_folder, label_file)
        missing_label_path = os.path.join(missing_labels_folder, label_file)
        shutil.move(label_path, missing_label_path)

# Move missing labels
for base_name in missing_in_labels:
    image_file = f"{base_name}.jpg"  # Assuming image files have .jpg extension
    if image_file in images_files:
        image_path = os.path.join(images_folder, image_file)
        missing_image_path = os.path.join(missing_images_folder, image_file)
        shutil.move(image_path, missing_image_path)

print("Missing files moved successfully.")

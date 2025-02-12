import os
import shutil

# Paths to the directories
base_dir = "../../BATCH_08_01_2025"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

new_base_dir = "../../data"
extra_images_dir = os.path.join(new_base_dir, "images")
extra_labels_dir = os.path.join(new_base_dir, "labels")

missing_images_dir = os.path.join(base_dir, "missing_images")
missing_labels_dir = os.path.join(base_dir, "missing_labels")

# Ensure missing directories exist
os.makedirs(missing_images_dir, exist_ok=True)
os.makedirs(missing_labels_dir, exist_ok=True)

# Get filenames without extensions
image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))}
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))}

# Find missing files
missing_in_images = label_files - image_files
missing_in_labels = image_files - label_files

# Function to check and copy files
def check_and_copy(missing_files, source_dir, target_dir, extension):
    for file in missing_files:
        source_path = os.path.join(source_dir, f"{file}.{extension}")
        if os.path.exists(source_path):
            target_path = os.path.join(target_dir, f"{file}.{extension}")
            shutil.copy(source_path, target_path)
            print(f"Copied {file}.{extension} to {target_dir}")
        else:
            print(f"{file}.{extension} not found in {source_dir}")

# Print and process missing files in images
print(f"{len(missing_in_images)} files missing in images directory:")
if missing_in_images:
    for file in missing_in_images:
        print(f"{file}.txt")
    check_and_copy(missing_in_images, extra_images_dir, missing_images_dir, "jpg")

# Print and process missing files in labels
print(f"{len(missing_in_labels)} files missing in labels directory:")
if missing_in_labels:
    for file in missing_in_labels:
        print(f"{file}.jpg")
    check_and_copy(missing_in_labels, extra_labels_dir, missing_labels_dir, "txt")

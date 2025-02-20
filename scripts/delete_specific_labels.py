import os

# Path to your labels folder
labels_dir = "/home/qulith-jr/Desktop/QL/shoplifting_yolo/datasets/crowdhuman-640x640/labels"

for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(labels_dir, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Filter out lines where class ID is 0
        updated_lines = [line for line in lines if not line.startswith("1 ")]

        # Write back the updated labels (overwrite the file)
        with open(file_path, "w") as f:
            f.writelines(updated_lines)
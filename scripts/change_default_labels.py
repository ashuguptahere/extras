import os


def change_labels(yolo_label_folder):
    for file_name in os.listdir(yolo_label_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(yolo_label_folder, file_name)

            # Read the file and update the class ID
            with open(file_path, "r") as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                parts = line.split()
                if parts:  # Ensure the line is not empty
                    parts[0] = "9"  # Update class ID
                    updated_lines.append(" ".join(parts) + "\n")

            # Write the updated lines back to the file
            with open(file_path, "w") as file:
                file.writelines(updated_lines)


if __name__ == "__main__":
    # Path to the folder containing YOLO label files
    yolo_label_folder = "labels"

    # Change labels
    change_labels(yolo_label_folder)

import os
from tqdm import tqdm


def update_labels(label_dir):
    # Define old and new class mappings
    class_map = {
        0: None,
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
        7: None,
        8: None,
        9: None,
        10: None,
        11: None,
        12: None,
        13: None,
        14: None,
        15: None,
        16: 8,
    }

    for file in tqdm(os.listdir(label_dir)):
        if file.endswith(".txt"):
            file_path = os.path.join(label_dir, file)
            updated_lines = []

            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])

                    if class_id in class_map and class_map[class_id] is not None:
                        new_class_id = class_map[class_id]
                        updated_line = f"{new_class_id} " + " ".join(parts[1:])
                        updated_lines.append(updated_line)

            with open(file_path, "w") as f:
                f.write("\n".join(updated_lines))


def delete_empty_txt(label_dir):
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            txt_path = os.path.join(label_dir, file)
            if os.path.getsize(txt_path) == 0:
                print(f"Deleting: {txt_path}")
                os.remove(txt_path)


def delete_missing_pair(image_dir, label_dir):
    images_files = os.listdir(image_dir)
    labels_files = os.listdir(label_dir)

    img_files = {f.rsplit(".", 1)[0] for f in images_files if "." in f}
    txt_files = {f.rsplit(".", 1)[0] for f in labels_files if "." in f}

    # Find unpaired files
    unpaired_txt = txt_files - img_files
    unpaired_img = img_files - txt_files
    print(unpaired_txt, unpaired_img)

    delete_count = 0

    # Delete unpaired image files
    for filename in unpaired_img:
        # List all files in the folder
        for file in os.listdir(image_dir):
            # Check if the file name (without extension) matches
            if os.path.splitext(file)[0] == filename:
                file_path = os.path.join(image_dir, file)
                print(f"Deleting: {file_path}")
                delete_count += 1
                os.remove(file_path)

    print("Deleted files:", delete_count)

    # Delete unpaired label files
    for filename in unpaired_txt:
        txt_path = os.path.join(label_dir, filename + ".txt")
        print(f"Deleting: {txt_path}")
        os.remove(txt_path)


if __name__ == "__main__":
    base_location = "/home/qulith-jr/Downloads/Gunmen Dataset/"
    image_dir = os.path.join(base_location, "images")
    label_dir = os.path.join(base_location, "labels")
    update_labels(label_dir)
    delete_empty_txt(label_dir)
    delete_missing_pair(image_dir, label_dir)

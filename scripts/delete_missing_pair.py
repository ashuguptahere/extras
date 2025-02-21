import os


def delete_missing_pair(images_path, labels_path):
    images_files = os.listdir(images_path)
    labels_files = os.listdir(labels_path)

    img_files = {f.rsplit(".", 1)[0] for f in images_files if "." in f}
    txt_files = {f.rsplit(".", 1)[0] for f in labels_files if "." in f}

    # Find unpaired files
    unpaired_txt = txt_files - img_files
    unpaired_img = img_files - txt_files
    print(unpaired_txt, unpaired_img)

    # Delete unpaired image files
    for filename in unpaired_img:
        # List all files in the folder
        for file in os.listdir(images_path):
            # Check if the file name (without extension) matches
            if os.path.splitext(file)[0] == filename:
                file_path = os.path.join(images_path, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)

    # Delete unpaired label files
    for filename in unpaired_txt:
        txt_path = os.path.join(labels_path, filename + ".txt")
        print(f"Deleting: {txt_path}")
        os.remove(txt_path)


if __name__ == "__main__":
    base_location = "/home/qulith-jr/Downloads/Gunmen Dataset/"
    images_path = os.path.join(base_location, "images")
    labels_path = os.path.join(base_location, "labels")
    delete_missing_pair(images_path, labels_path)

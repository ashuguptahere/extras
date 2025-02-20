import os


def delete_missing_pair(folder):
    files = os.listdir(folder)
    txt_files = {f[:-4] for f in files if f.endswith(".txt")}
    img_files = {f[:-4] for f in files if f.endswith(".jpg")}

    # Find unpaired files
    unpaired_txt = txt_files - img_files
    unpaired_img = img_files - txt_files
    print(unpaired_txt, unpaired_img)

    # Delete unpaired files
    for filename in unpaired_txt:
        txt_path = os.path.join(folder, filename + ".txt")
        print(f"Deleting: {txt_path}")
        os.remove(txt_path)

    for filename in unpaired_img:
        img_path = os.path.join(folder, filename + ".jpg")
        print(f"Deleting: {img_path}")
        os.remove(img_path)


folder_path = "images"
delete_missing_pair(folder_path)

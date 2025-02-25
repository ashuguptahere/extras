import os
from tqdm import tqdm

# Define old and new class mappings
class_map = {
    0: 0,  # carry/hold
    1: None,  # crawl (remove)
    2: 1,  # crouch/kneel
    3: 2,  # grab
    4: None,  # jump
    5: None,  # kick
    6: None,  # run (remove)
    7: 3,  # shoplift
    8: None,  # sit (remove)
    9: 4,  # stand
    10: 5,  # talk
    11: 6  # walk

    # # Reverse class mapping
    # 0: 0,
    # 1: 2,
    # 2: 3,
    # 3: 7,
    # 4: 9,
    # 5: 10,
    # 6: 11,
}

def update_labels(label_dir):
    for file in tqdm(os.listdir(label_dir)):
        if file.endswith('.txt'):
            file_path = os.path.join(label_dir, file)
            updated_lines = []
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    
                    if class_id in class_map and class_map[class_id] is not None:
                        new_class_id = class_map[class_id]
                        updated_line = f"{new_class_id} " + " ".join(parts[1:])
                        updated_lines.append(updated_line)
            
            with open(file_path, 'w') as f:
                f.write("\n".join(updated_lines))

# Update labels in your dataset
label_directory = "labels_new/"  # Replace with your label folder path
update_labels(label_directory)

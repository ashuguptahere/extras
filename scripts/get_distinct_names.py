import os

def get_distinct_filenames(directory):
    distinct_names = set()
    
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if "_frame" in filename:
            base_name = filename.split("_frame")[0]  # Extract part before "_frame"
            distinct_names.add(base_name)
    
    return distinct_names

def main():
    images_dir = "/home/qulith-jr/Desktop/QL/datasets/cleaned-and-combined/images"  # Change this to your directory path if needed
    
    if not os.path.exists(images_dir):
        print(f"Directory '{images_dir}' does not exist.")
        return
    
    distinct_filenames = get_distinct_filenames(images_dir)
    print("Distinct Filenames:", sorted(distinct_filenames))
    print("Length:", len(distinct_filenames))

if __name__ == "__main__":
    main()
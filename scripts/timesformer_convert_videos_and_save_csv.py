import os
import csv
import random
import subprocess
import concurrent.futures


def convert_videos(input_dir, output_dir, resolution=(224, 224)):
    """Convert all videos in input_dir to .mp4 format with a fixed resolution and save in output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_filename = (
            os.path.splitext(filename)[0] + ".mp4"
        )  # Ensure output is .mp4
        output_path = os.path.join(output_dir, output_filename)

        if not os.path.isfile(input_path):
            continue  # Skip directories

        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            f"scale={resolution[0]}:{resolution[1]}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            output_path,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Converted: {input_path} -> {output_path}")
        except subprocess.CalledProcessError:
            print(f"Failed to convert: {input_path}")


def get_video_files(directory, label):
    """Retrieve absolute paths of all files in the given directory with their label."""
    return [
        (os.path.abspath(os.path.join(root, file)), label)
        for root, _, files in os.walk(directory)
        for file in files
    ]


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split data into train, validation, and test sets."""
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def save_csv(base_path, filename, data):
    """Save data to a CSV file in the base path."""
    filepath = os.path.join(base_path, filename)
    with open(filepath, "w", newline="") as csvfile:
        csv.writer(csvfile, delimiter=" ").writerows(data)
    print(f"CSV file '{filepath}' created successfully.")


def create_splits(base_path, directories):
    """Create train, val, and test CSV files from multiple directories."""
    data = [
        file
        for idx, dir in enumerate(directories)
        for file in get_video_files(dir, idx)
    ]
    train_data, val_data, test_data = split_data(data)
    for name, dataset in zip(
        ["train.csv", "val.csv", "test.csv"], [train_data, val_data, test_data]
    ):
        save_csv(base_path, name, dataset)


if __name__ == "__main__":
    base_input = "/home/qulith-jr/Desktop/QL/datasets/KTH_dataset"
    base_output = "/home/qulith-jr/Desktop/QL/datasets/KTH_dataset_224x224"
    categories = [
        "boxing",
        "handclapping",
        "handwaving",
        "jogging",
        "normal",
        "running",
        "shoplifting",
        "walking",
    ]

    # with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    #     futures = [
    #         executor.submit(
    #             convert_videos,
    #             os.path.join(base_input, category),
    #             os.path.join(base_output, category),
    #         )
    #         for category in categories
    #     ]
    #     concurrent.futures.wait(futures)

    create_splits(base_output, [os.path.join(base_output, cat) for cat in categories])

import os
import csv
import cv2
import glob
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from ultralytics import YOLO


# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def load_images_and_labels(image_dir, label_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    return image_paths, label_paths


def read_yolo_labels(label_path):
    bboxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            values = line.strip().split()
            label = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
            bboxes.append((label, x_center, y_center, width, height))
    return bboxes


def yolo_to_bbox(image_shape, yolo_bbox):
    label, x_center, y_center, width, height = yolo_bbox
    h, w = image_shape[:2]
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)
    return label, x1, y1, x2, y2


def perform_pose_estimation(model, image, bbox):
    x1, y1, x2, y2 = bbox
    cropped_img = image[y1:y2, x1:x2]

    # # Run pose estimation using ultralytics
    # results = model(cropped_img, verbose=False)
    # keypoints_list = []

    # for result in results:
    #     for kp in result.keypoints.xy.cpu().numpy():
    #         keypoints_list.append(kp.tolist())

    # return list(np.array(keypoints_list).flat)

    # Run pose estimation using mediapipe
    keypoints_list = []
    image_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])  # Normalize coordinates

        keypoints_list.append(keypoints)
    else:
        return None

    return list(np.array(keypoints_list).flat)


def save_to_csv(data, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image", "label", "x1", "y1", "x2", "y2"]
            + [f"k_x{i}" for i in range(33)]
            + [f"k_y{i}" for i in range(33)]
            + [f"k_z{i}" for i in range(33)]
        )
        for row in data:
            writer.writerow(row)


def main(image_dir, label_dir, output_file):
    image_paths, label_paths = load_images_and_labels(image_dir, label_dir)
    model = YOLO("yolo11x-pose.pt", verbose=False)
    pose_data = []

    for image_path, label_path in tqdm(zip(image_paths, label_paths)):
        image = cv2.imread(image_path)
        bboxes = read_yolo_labels(label_path)

        for bbox in bboxes:
            label, x1, y1, x2, y2 = yolo_to_bbox(image.shape, bbox)
            keypoints = perform_pose_estimation(model, image, (x1, y1, x2, y2))
            if keypoints is None:
                continue
            pose_data.append(
                [os.path.basename(image_path), label, x1, y1, x2, y2] + keypoints
            )

    save_to_csv(pose_data, output_file)
    print(f"Pose data saved to {output_file}")


if __name__ == "__main__":
    # For val dataset
    base_dir = "/home/qulith-jr/Desktop/QL/datasets/dataset/val/"
    image_dir = os.path.join(base_dir, "images")  # Folder containing images
    label_dir = os.path.join(base_dir, "labels")  # Folder containing labels
    main(image_dir, label_dir, "val_pose_data.csv")

    # For test dataset
    base_dir = "/home/qulith-jr/Desktop/QL/datasets/dataset/test/"
    image_dir = os.path.join(base_dir, "images")  # Folder containing images
    label_dir = os.path.join(base_dir, "labels")  # Folder containing labels
    main(image_dir, label_dir, "test_pose_data.csv")

    # For train dataset
    base_dir = "/home/qulith-jr/Desktop/QL/datasets/dataset/train/"
    image_dir = os.path.join(base_dir, "images")  # Folder containing images
    label_dir = os.path.join(base_dir, "labels")  # Folder containing labels
    main(image_dir, label_dir, "train_pose_data.csv")

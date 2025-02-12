import os
import shutil
import pandas as pd
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def build_dataframe(yolo_label_folder):
    """
    Converts the .txt files data into pandas dataframe.
    Parameters:
        yolo_label_folder (str): Path to the folder containing YOLO label files.
    Returns:
        pd.DataFrame: A pandas dataframe with 6 columns, i.e, filename, class_id, center_x, center_y, width and height.
    """
    data = []
    for filename in os.listdir(yolo_label_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(yolo_label_folder, filename), "r") as file:
                for line in file:
                    element = deque(line.strip().split(" "))
                    element.appendleft(filename.split(".txt")[0])
                    data.append(element)
    return pd.DataFrame(
        data,
        columns=["filename", "class_id", "center_x", "center_y", "width", "height"],
    )


def plot_bar_chart(df, labels, figname):
    """
    Plot the bar chart.
    Parameters:
        df (pd.DataFrame): the dataframe obtained from build_dataframe(method).
    """

    # Map class_id (0-11) to labels
    id_to_label = {i + 1: labels[i] for i in range(len(labels))}

    # Add new column with labels
    df["class_label"] = df["class_id"].map(id_to_label)

    # Count occurrences of each class_label
    class_counts = df["class_label"].value_counts().reindex(labels, fill_value=0)

    # Plot
    plt.figure(figsize=(15, 10))
    bars = plt.bar(class_counts.index, class_counts, color="skyblue", edgecolor="black")

    # Annotate bars with counts
    for bar, count in zip(bars, class_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
            bar.get_height() + 0.2,  # Y position (above bar)
            str(int(count)),  # Convert count to string
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Labels and styling
    plt.xlabel("Action Class")
    plt.ylabel("Count")
    plt.title("Action Class Distribution")
    plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(figname)
    plt.show()


def ava_format_data_creation(images_dir, rawframes_dir):
    df = pd.DataFrame(os.listdir(images_dir), columns=["filename"])

    df["filename_prefix"] = df["filename"].apply(lambda x: x.split(".jpg")[0][:-12])

    # Get unique prefixes
    unique_prefixes = df["filename_prefix"].unique()

    # Create the 'rawframes' directory if it doesn't exist
    if not os.path.exists(rawframes_dir):
        os.makedirs(rawframes_dir)

    # Create a folder for each unique prefix inside 'rawframes'
    for prefix in unique_prefixes:
        folder_path = os.path.join(rawframes_dir, prefix)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    df["filename_prefix"] = df["filename"].apply(lambda x: x.split(".jpg")[0][:-12])
    print("Number of unique folders:", len(unique_prefixes))

    for index, row in tqdm(df.iterrows()):
        filename = row["filename"]
        prefix = row["filename_prefix"]

        # Construct the source and destination paths
        src_path = os.path.join(images_dir, filename)
        dest_path = os.path.join(rawframes_dir, prefix, filename)

        # Copy the file if it exists
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            # print(f"Copied {filename} to {dest_path}")
        else:
            print(f"File {filename} does not exist in the 'images' directory.")

    return set(unique_prefixes)


def remove_unused_labels_and_reorganise_classes(df):
    df = df[df["class_id"] != 2]
    df = df[df["class_id"] != 5]
    df = df[df["class_id"] != 6]
    df = df[df["class_id"] != 7]
    df = df[df["class_id"] != 9]
    df = df.reset_index(drop=True)

    df.loc[df["class_id"] == 3, "class_id"] = 2
    df.loc[df["class_id"] == 4, "class_id"] = 3
    df.loc[df["class_id"] == 8, "class_id"] = 4
    df.loc[df["class_id"] == 10, "class_id"] = 5
    df.loc[df["class_id"] == 11, "class_id"] = 6
    df.loc[df["class_id"] == 12, "class_id"] = 7

    return df


def save_annotation_files(df):
    # Saving ava_test_v2.1.txt file
    test = pd.DataFrame(os.listdir("rawframes"))
    test.to_csv("annotations/ava_test_v2.1.txt", index=False, header=False)

    # Splitting the data
    X = df.drop("class_id", axis=1)
    y = df[["class_id"]]

    # Split into train and validation sets without shuffling, but still stratified
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Combine features and target for the training set
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data = train_data.sort_values(by=["filename_prefix", "timestamp"])
    train_data.to_csv("annotations/ava_train_v2.1.csv", index=False, header=False)

    # Combine features and target for the validation/test set
    val_data = pd.concat([X_val, y_val], axis=1)
    val_data = val_data.sort_values(by=["filename_prefix", "timestamp"])
    val_data.to_csv("annotations/ava_val_v2.1.csv", index=False, header=False)

    print(
        "Files saved as 'annotations/ava_train_v2.1.csv', 'annotations/ava_test_v2.1.txt' and 'annotations/ava_val_v2.1.csv'"
    )


if __name__ == "__main__":
    # Path to the folder containing YOLO image/label files
    images_dir = "../../Combined/images"
    labels_dir = "../../Combined/labels"
    rawframes_dir = "rawframes"

    # Build Pandas DataFrame
    df = build_dataframe(labels_dir)
    
    df["person_id"] = 0

    # Convert "class_id" column of string type to int
    df["class_id"] = df["class_id"].astype(int)

    # Adding 1 to every row of the class_id column
    df["class_id"] = df["class_id"] + 1

    # Define labels mapping
    labels = [
        "carry/hold",
        "crawl",
        "crouch/kneel",
        "grab",
        "jump",
        "kick",
        "run",
        "shoplift",
        "sit",
        "stand",
        "talk",
        "walk",
    ]

    # Plot bar chart of the dataframe's class_label
    plot_bar_chart(df, labels, figname="fig.png")

    # Extract the numerical part
    df["number"] = df["filename"].apply(lambda x: x.split(".jpg")[0][-5:]).astype(int)

    # Map the numbers to timestamps
    df["timestamp"] = 900 + df["number"].values

    df = df.sort_values(by="filename")

    # Extract the prefix before the last "_number"
    df["filename_prefix"] = df["filename"].apply(lambda x: x.split(".jpg")[0][:-12])

    df = df[
        [
            "filename_prefix",
            "timestamp",
            "center_x",
            "center_y",
            "width",
            "height",
            "class_id",
            "person_id",
        ]
    ]

    # Create rawframes folder and move every images to the respective folder
    ava_format_data_creation(images_dir, rawframes_dir)

    # Remove unused labels
    df = remove_unused_labels_and_reorganise_classes(df)

    # Save the dataframe to csv file
    df.to_csv("data.csv", index=False, header=False)
    print("Successfully saved data.csv!")

    # Save annotation files
    save_annotation_files(df)

    # Define labels mapping
    labels = [
        "carry/hold",
        "crouch/kneel",
        "grab",
        "shoplift",
        "stand",
        "talk",
        "walk",
    ]

    plot_bar_chart(df, labels, figname="fig2.png")

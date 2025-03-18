import os
import pandas as pd
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt


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


def plot_bar_chart(df, labels):
    figname = datetime.now().strftime("figs/fig_%Y%m%d_%H%M%S.png")
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
    print(class_counts.values)

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


if __name__ == "__main__":
    # Path to the folder containing YOLO image/label files
    base_location = "../../new/val/"
    images_dir = os.path.join(base_location, "images")
    labels_dir = os.path.join(base_location, "labels")

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
        "crouch/kneel",
        "grab",
        "shoplift",
        "stand",
        "talk",
        "walk",
        "violence",
        "weapon",
        "face",
    ]

    # Plot bar chart of the dataframe's class_label
    plot_bar_chart(df, labels)

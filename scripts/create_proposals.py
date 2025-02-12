import pickle
import pandas as pd

# Load the CSV files (assume no end time and no confidence)
train_df = pd.read_csv(
    "annotations/ava_train_v2.1.csv",
    header=None,
    names=["video_id", "timestamp", "action_id"],
)
val_df = pd.read_csv(
    "annotations/ava_val_v2.1.csv",
    header=None,
    names=["video_id", "timestamp", "action_id"],
)


# Function to create proposals with fixed duration
def create_proposals(df, duration=1.0):
    proposals = {}

    for _, row in df.iterrows():
        video_id = row["video_id"]
        start_time = row["timestamp"]
        action_id = int(row["action_id"])  # Ensure action_id is an integer

        # Assuming end time by adding a fixed duration to the start time
        end_time = start_time + duration

        # Create proposal structure with the given start and end times
        proposal = {
            "proposal": [start_time, end_time],
            "score": 1.0,  # Set confidence to 1.0 as default (or another default value)
            "class_probs": [0.0]
            * 7,  # 7 action classes for AVA (set all probabilities to 0 initially)
        }

        # Set the probability for the specific action class (1.0 for the correct class)
        proposal["class_probs"][action_id - 1] = (
            1.0  # action_id starts from 1, so subtract 1
        )

        # Append proposal to the dictionary for this video
        if video_id not in proposals:
            proposals[video_id] = []
        proposals[video_id].append(proposal)

    return proposals


# Generate proposals for training and validation sets
train_proposals = create_proposals(train_df)
val_proposals = create_proposals(val_df)

# Save proposals to .pkl files
with open("annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl", "wb") as f:
    pickle.dump(train_proposals, f)

with open("annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl", "wb") as f:
    pickle.dump(val_proposals, f)

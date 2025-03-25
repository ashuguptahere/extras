import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO


def process_video(video_path):
    # Load the YOLO pose estimation model
    model = YOLO("yolo11x-pose.pt")

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose estimation
        results = model.track(frame, persist=True, verbose=False)
        try:
            person_id = int(results[0].boxes.id.tolist()[0])
        # except:
        #     print("error with person id")

        # annotated_frame = results[0].plot()

        # # Display the frame with keypoints
        # cv2.imshow("Pose Estimation", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

            # Extract keypoints from results
            for result in results:
                if result.keypoints is not None and result.keypoints.xy is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    confidences = (
                        result.keypoints.conf.cpu().numpy()
                        if result.keypoints.conf is not None
                        else None
                    )

                    # Flatten keypoints and confidences
                    flattened_data = []
                    for i in range(17):
                        if i < len(keypoints[0]):
                            conf_value = (
                                confidences[0][i] if confidences is not None else None
                            )
                            # instead of xy coordinates we are storing yx coordinates as per poselift
                            flattened_data += [
                                keypoints[0][i][1],
                                keypoints[0][i][0],
                                conf_value,
                            ]

                            # # Draw keypoints on the frame - Green dot for keypoints
                            # x, y = int(keypoints[0][i][0]), int(keypoints[0][i][1])
                            # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        else:
                            # If keypoints are missing
                            flattened_data += [
                                None,
                                None,
                                None,
                            ]

                    flattened_data = np.array(flattened_data, dtype=np.float32).tolist()

                    # Ensure person exists in dictionary
                    if person_id not in poselift_data:
                        poselift_data[person_id] = {}

                    # Store frame data
                    poselift_data[person_id][frame_count] = {
                        "keypoints": [flattened_data],
                        "scores": None,
                    }

                    # # Display the frame with keypoints
                    # cv2.imshow("Pose Estimation", frame)
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     break

            frame_count += 1
        except Exception as e:
            print("error with person id", e)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # for video in tqdm(sorted(glob("videos/*"))):
        video = "videos/0011.mp4"
        filename = video.split("/")[1].split(".")[0]

        # Dictionary to store PoseLift data
        poselift_data = {}
        output_json_path = f"outputs2/{filename}_data.json"
        process_video(video)

        # Save PoseLift JSON file
        with open(output_json_path, "w") as json_file:
            json.dump(poselift_data, json_file, indent=4)
        print(f"Processing complete. Data saved to {output_json_path}")

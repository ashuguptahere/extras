import os
import cv2


def display_video_with_bboxes(video_path, bboxes_dir):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Generate the corresponding bounding box file name
        txt_name = f"frame_{frame_idx}.txt"
        txt_path = os.path.join(bboxes_dir, txt_name)

        # Check if bounding box file exists
        if os.path.exists(txt_path):
            # Read the bounding boxes from the file and draw them on the frame
            with open(txt_path, "r") as f:
                for line in f:
                    # Each line contains x_min, y_min, x_max, y_max coordinates
                    x_min, y_min, x_max, y_max = map(float, line.strip().split())

                    # Draw the bounding box on the frame
                    cv2.rectangle(
                        frame,
                        (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        (0, 255, 0),
                        2,
                    )  # Green box with thickness 2

        # Display the frame with bounding boxes
        cv2.imshow("Video with Person Detection", frame)

        # Press 'q' to exit the video display
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        # Increment the frame index
        frame_idx += 1

    # Release the video capture object and close display window
    cap.release()
    cv2.destroyAllWindows()


# Usage
display_video_with_bboxes("videos/IP Camera4_L0524 WRM COLOMBO I_L0524 WRM COLOMBO I_20240509213456_20240509213959_89967941.mp4", "labels")

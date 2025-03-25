import os
import cv2

def images_to_video(image_folder, output_video, fps=25):
    # Get all image files from the folder
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    images.sort()  # Sort files to maintain order
    
    if not images:
        print("No images found in the directory.")
        return
    
    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Skipping invalid image: {image}")
            continue
        
        video.write(frame)
    
    video.release()
    print(f"Video saved as {output_video}")

# Usage
images_to_video('images2', 'output_video.mp4', fps=25)

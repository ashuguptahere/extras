import os
import glob
import subprocess
import concurrent.futures


def get_total_frames(video_path):
    """Get the total number of frames in a video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    return int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0


def split_video(video_path, output_dir, segment_length=8):
    """Splits a video into multiple parts, each containing `segment_length` frames."""
    total_frames = get_total_frames(video_path)
    if total_frames < segment_length:
        print(f"Skipping {video_path}, not enough frames.")
        return

    video_name, ext = os.path.splitext(os.path.basename(video_path))
    num_segments = total_frames // segment_length  # Get max number of segments

    for i in range(num_segments):
        start_frame = i * segment_length
        output_file = os.path.join(output_dir, f"{video_name}_part_{i + 1}{ext}")

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"select='between(n\,{start_frame}\,{start_frame + segment_length - 1})'",
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            output_file,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Created: {output_file}")


def process_videos(input_folder, output_folder, segment_length=8):
    """Process multiple videos in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    video_files = glob.glob(
        os.path.join(input_folder, "*.mp4")
    )  # Adjust extension if needed

    for video_file in video_files:
        split_video(video_file, output_folder, segment_length)


if __name__ == "__main__":
    base_input = "/home/qulith-jr/Desktop/QL/datasets/KTH_dataset_224x224"
    output_folder = "/home/qulith-jr/Desktop/QL/datasets/KTH_dataset_224x224_8frame"
    categories = [
        d for d in os.listdir(base_input) if os.path.isdir(os.path.join(base_input, d))
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [
            executor.submit(
                process_videos,
                os.path.join(base_input, category),
                os.path.join(output_folder, category),
            )
            for category in categories
        ]
        concurrent.futures.wait(futures)

    process_videos(base_input, output_folder, segment_length=8)

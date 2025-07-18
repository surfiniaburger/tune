# process_video.py
import cv2
import os
from pathlib import Path

# --- CONFIGURATION ---
# Put your video files in a folder named 'videos' inside your project directory
VIDEO_SOURCE_DIR = Path("videos")
# The script will save the extracted frames here
FRAME_OUTPUT_DIR = Path("extracted_frames")

# This is the key setting: Save one frame every N frames.
# A video is often 30 frames per second. A value of 30 means "save one picture per second."
# A good starting point is 15 (two pictures per second).
FRAME_CAPTURE_INTERVAL = 15 

def extract_frames_from_videos():
    """
    Loops through all videos in the source directory and extracts frames
    at a specified interval.
    """
    print("🚀 Starting video frame extraction...")
    if not VIDEO_SOURCE_DIR.exists():
        print(f"❌ Error: Source directory '{VIDEO_SOURCE_DIR}' not found.")
        print("Please create it and place your video files inside.")
        return

    FRAME_OUTPUT_DIR.mkdir(exist_ok=True)
    video_files = list(VIDEO_SOURCE_DIR.glob("*.*")) # Find all files

    if not video_files:
        print(f"🤷 No video files found in '{VIDEO_SOURCE_DIR}'.")
        return

    for video_path in video_files:
        print(f"\nProcessing video: {video_path.name}")
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ Error opening video file: {video_path}")
                continue

            frame_count = 0
            saved_count = 0
            while True:
                # Read one frame from the video
                success, frame = cap.read()
                if not success:
                    break # End of video

                # Check if this frame is one we should save
                if frame_count % FRAME_CAPTURE_INTERVAL == 0:
                    # Construct a unique filename
                    video_name = video_path.stem
                    output_filename = f"{video_name}_frame_{saved_count:04d}.jpg"
                    output_path = FRAME_OUTPUT_DIR / output_filename
                    
                    # Save the frame as a JPG image
                    cv2.imwrite(str(output_path), frame)
                    saved_count += 1
                
                frame_count += 1

            cap.release()
            print(f"✅ Finished. Saved {saved_count} frames from {video_path.name}.")

        except Exception as e:
            print(f"❌ An error occurred processing {video_path.name}: {e}")

    print("\n🎉 All videos processed!")


if __name__ == "__main__":
    extract_frames_from_videos()
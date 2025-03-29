import cv2
import os

# Path to the videos folder and where to save extracted frames
video_folder = r"D:\AI&ML&DEEP_LEARNING\datasetforsign\videos"
output_folder = r"D:\AI&ML&DEEP_LEARNING\datasetforsign\extracted_frames"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def extract_frames(video_path, output_folder, video_name):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"{video_name}_frame_{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()


for video_file in os.listdir(video_folder):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        extract_frames(video_path, output_folder, video_name)

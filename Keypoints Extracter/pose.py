import cv2
import mediapipe as mp
import os
import pandas as pd
import json
import numpy as np

# Initialize MediaPipe Pose and Hands solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Folder with the extracted video frames
frame_folder = r"D:\AI&ML&DEEP_LEARNING\datasetforsign\test"
output_folder = r"D:\AI&ML&DEEP_LEARNING\datasetforsign\pose_keypoints"
visualizations_folder = os.path.join(output_folder, "visualizations")

# Create output folders if they don't exist
for folder in [output_folder, visualizations_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def extract_and_visualize_keypoints(image, frame_file):
    # Convert the frame to RGB as MediaPipe expects RGB images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract pose keypoints
    pose_results = pose.process(image_rgb)
    hand_results = hands.process(image_rgb)
    
    keypoints = {}
    
    # Create a copy of the image for visualization
    visualization_img = image.copy()
    
    # Extract and visualize pose landmarks if available
    if pose_results.pose_landmarks:
        keypoints['pose'] = [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark]
        mp_drawing.draw_landmarks(
            visualization_img, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))
    
    # Extract and visualize hand landmarks if available (both hands)
    if hand_results.multi_hand_landmarks:
        keypoints['hands'] = []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            keypoints['hands'].append([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            mp_drawing.draw_landmarks(
                visualization_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=2))
    
    # Save the visualization image
    vis_path = os.path.join(visualizations_folder, f"vis_{frame_file}")
    cv2.imwrite(vis_path, visualization_img)
    
    return keypoints if keypoints else None, vis_path

# Process each frame and extract keypoints
keypoint_data = []
for frame_file in os.listdir(frame_folder):
    if frame_file.endswith('.jpg'):
        frame_path = os.path.join(frame_folder, frame_file)
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to read image: {frame_path}")
                continue
            
            keypoints, vis_path = extract_and_visualize_keypoints(frame, frame_file)
            
            if keypoints:
                # Save keypoints and frame information
                keypoint_data.append({
                    'frame': frame_file,
                    'keypoints': json.dumps(keypoints),  # Convert dict to JSON string
                    'visualization': vis_path
                })
            else:
                print(f"No keypoints found for frame: {frame_file}")
        except Exception as e:
            print(f"Error processing frame {frame_file}: {str(e)}")

# Save the extracted keypoints to a CSV file
try:
    keypoints_df = pd.DataFrame(keypoint_data)
    output_csv = os.path.join(output_folder, 'pose_keypoints.csv')
    keypoints_df.to_csv(output_csv, index=False)
    print(f"Keypoints saved to {output_csv}")
except Exception as e:
    print(f"Error saving keypoints to CSV: {str(e)}")

print(f"Processed {len(keypoint_data)} frames")
print(f"Visualizations saved in {visualizations_folder}")

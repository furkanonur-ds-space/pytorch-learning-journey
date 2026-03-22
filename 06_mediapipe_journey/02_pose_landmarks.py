# Topic: Human Pose Estimation using MediaPipe Pose Landmarker
# Source: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
# Summary:
#   1. Detect 33 3D body landmarks (shoulders, elbows, knees, etc.) from an image.
#   2. Download the 'pose_landmarker_heavy.task' model.
#   3. Initialize the PoseLandmarker API.
#   4. Run inference to get body landmarks.
#   5. Draw the full-body skeleton on the image using mp.solutions.drawing_utils.
#
# Key Concept:
#   MediaPipe Pose Landmarker tracks 33 topological landmarks across the human body
#   in both 2D image coordinates and 3D real-world coordinates. This is heavily
#   used in fitness apps, gesture control, and motion capture.

import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw the 33 body landmarks and their connecting lines (the skeleton)."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses (usually 1 person per image, but can be multiple)
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


def main():
    print("\n=======================================================")
    print("  MediaPipe: Pose Landmarker (Static Image)")
    print("=======================================================")

    # -------------------------------------------------------
    # 1. DOWNLOAD THE MODEL AND TEST IMAGE
    # -------------------------------------------------------
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        print(f"[*] Downloading MediaPipe Pose model ({model_path})...")
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
            out_file.write(response.read())
        print("[OK] Model downloaded.")
        
    img_path = 'yoga_pose.jpg'
    if not os.path.exists(img_path):
        print(f"[*] Downloading sample image...")
        # Using a public domain image
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
            out_file.write(response.read())
        print("[OK] Image downloaded.")


    # -------------------------------------------------------
    # 2. CREATE THE TASK OPTIONS
    # -------------------------------------------------------
    print("\n[*] Initializing Pose Landmarker...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1  # Maximum number of poses/people to detect
    )

    # -------------------------------------------------------
    # 3. RUN INFERENCE
    # -------------------------------------------------------
    detector = vision.PoseLandmarker.create_from_options(options)

    # Load image using MediaPipe's utilities
    mp_image = mp.Image.create_from_file(img_path)

    print("[*] Extracting 33 body landmarks...")
    detection_result = detector.detect(mp_image)
    
    # -------------------------------------------------------
    # 4. EXAMINE THE RESULTS
    # -------------------------------------------------------
    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")
    poses_found = len(detection_result.pose_landmarks)
    print(f"  People found: {poses_found}\n")

    if poses_found > 0:
        first_person = detection_result.pose_landmarks[0]
        print(f"  Extracted exactly {len(first_person)} anatomical keypoints.")
        
        # Landmark indices reference:
        # 0: Nose, 11: Left Shoulder, 12: Right Shoulder
        # 15: Left Wrist, 16: Right Wrist, 23: Left Hip, 24: Right Hip
        nose = first_person[0]
        print(f"  Nose coordinates (Relative) : X: {nose.x:.3f}, Y: {nose.y:.3f}")
        print(f"  MediaPipe also provides 3D coordinates (Z metric) representing depth!")

    # -------------------------------------------------------
    # 5. VISUALIZE AND SAVE
    # -------------------------------------------------------
    rgb_data = mp_image.numpy_view()
    
    # Draw the skeleton on the RGB image
    annotated_rgb = draw_landmarks_on_image(rgb_data, detection_result)
    
    # Convert RGB to BGR for OpenCV saving
    bgr_annotated = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, '02_pose_landmarks_result.jpg')
    cv2.imwrite(out_path, bgr_annotated)
    
    print(f"\n  Saved visualization to: {out_path}")
    print("  ✅ Script completed successfully.\n")

if __name__ == '__main__':
    main()

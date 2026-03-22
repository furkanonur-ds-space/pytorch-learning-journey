# Topic: Real-Time Face and Pose Detection
# Source: MediaPipe Vision Live Stream documentation
# Summary:
#   1. Capture real-time video frames from your computer's webcam using OpenCV.
#   2. Asynchronously combine both FaceDetector and PoseLandmarker models.
#   3. Send every video frame to the models in LIVE_STREAM mode.
#   4. Use a callback function to receive and draw landmarks on the fly.
#   5. Press 'q' to quit the live feed.
#
# Key Concept:
#   MediaPipe's greatest strength is CPU efficiency. By using REAL-TIME asynchronous
#   streaming (`running_mode=LIVE_STREAM`), we get zero lag between the camera 
#   capturing a frame and the ML models overlaying the skeleton on your body.

import os
import cv2
import time
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np

# Global variables to store the latest asynchronous results
latest_face_result = None
latest_pose_result = None


# ---------------------------------------------------------
# CALLBACK FUNCTIONS
# ---------------------------------------------------------
# In LIVE_STREAM mode, MediaPipe models process frames in the background
# and return the results to these callbacks immediately when ready.

def face_result_callback(result: vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global latest_face_result
    latest_face_result = result


def pose_result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_pose_result
    latest_pose_result = result


# ---------------------------------------------------------
# DRAWING HELPERS
# ---------------------------------------------------------

def draw_faces(image_bgr, detection_result):
    if not detection_result or not detection_result.detections:
        return image_bgr
        
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image_bgr, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(image_bgr, 'Face', (bbox.origin_x, bbox.origin_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image_bgr


def draw_pose(image_bgr, detection_result):
    if not detection_result or not detection_result.pose_landmarks:
        return image_bgr
        
    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=image_bgr,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
        )
    return image_bgr


def download_models_if_needed():
    print("[*] Verifying models...")
    models = {
        'blaze_face_short_range.tflite': 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
        'pose_landmarker_heavy.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    }
    for file_name, url in models.items():
        if not os.path.exists(file_name):
            print(f"    Downloading {file_name}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(file_name, 'wb') as out_file:
                out_file.write(response.read())
    print("    Models are ready.\n")


def main():
    print("\n=======================================================")
    print("  MediaPipe: Real-Time Face & Pose Tracking (Webcam)")
    print("=======================================================")
    print("  Press 'q' in the video window to stop.")
    
    download_models_if_needed()

    # -------------------------------------------------------
    # 1. SETUP BOTH MODELS FOR LIVE STREAMING
    # -------------------------------------------------------
    # Face Options
    face_options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=face_result_callback
    )
    
    # Pose Options
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=pose_result_callback
    )

    # -------------------------------------------------------
    # 2. OPEN WEBCAM
    # -------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Use a context manager to ensure models are closed cleanly
    with vision.FaceDetector.create_from_options(face_options) as face_detector, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_detector:

        print("\n[*] Camera started! Looking for faces and bodies...")
        
        while cap.isOpened():
            success, frame_bgr = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Open CV reads frames as BGR, but MediaPipe expects RGB.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # MediaPipe Image Object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Send the frame to the models.
            # We must pass a timestamp in milliseconds.
            timestamp_ms = int(time.time() * 1000)
            
            # These are un-blocking asynchronous calls!
            # The result goes to the callbacks above.
            face_detector.detect_async(mp_image, timestamp_ms)
            pose_detector.detect_async(mp_image, timestamp_ms)

            # -------------------------------------------------------
            # 3. DRAW LATEST ASYNC RESULTS
            # -------------------------------------------------------
            annotated_frame = frame_bgr.copy()
            
            # Overlay Skeleton & Bounding Boxes
            annotated_frame = draw_pose(annotated_frame, latest_pose_result)
            annotated_frame = draw_faces(annotated_frame, latest_face_result)

            # Flip the image horizontally for a selfie-view display.
            # This makes moving left/right feel natural instead of inverted.
            annotated_frame = cv2.flip(annotated_frame, 1)

            cv2.imshow('MediaPipe Real-Time Tracking', annotated_frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("\n  Closing webcam stream. Goodbye!")
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

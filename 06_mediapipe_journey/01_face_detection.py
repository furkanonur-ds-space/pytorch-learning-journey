# Topic: Face Detection using Google MediaPipe
# Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
# Summary:
#   1. MediaPipe is a cross-platform, ultra-fast ML framework by Google.
#   2. We download the 'blaze_face_short_range.tflite' model.
#   3. We use the FaceDetector Task API to find faces in an image.
#   4. For each face, we get a bounding box and 6 key landmarks:
#      (left eye, right eye, nose tip, mouth, left ear tragion, right ear tragion).
#   5. Draw the results using MediaPipe's built-in drawing utilities.
#
# Key Concept:
#   Unlike heavy PyTorch models (like Faster R-CNN), MediaPipe uses highly
#   optimized TFLite models designed to run in real-time on CPUs and mobile devices.

import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


def draw_faces_on_image(rgb_image, detection_result):
    """Draw bounding boxes and 6 key facial landmarks on the image."""
    # We create a copy to draw on
    annotated_image = np.copy(rgb_image)
    
    for detection in detection_result.detections:
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 2)

        # Draw 6 key facial landmarks
        # MediaPipe provides landmarks as relative coordinates [0.0, 1.0].
        # We need to multiply them by image dimensions to get actual pixels.
        for keypoint in detection.keypoints:
            x = int(keypoint.x * rgb_image.shape[1])
            y = int(keypoint.y * rgb_image.shape[0])
            cv2.circle(annotated_image, (x, y), 3, (0, 0, 255), -1)

    return annotated_image


def main():
    print("\n=======================================================")
    print("  MediaPipe: Face Detection (Static Image)")
    print("=======================================================")

    # -------------------------------------------------------
    # 1. DOWNLOAD THE MODEL AND TEST IMAGE
    # -------------------------------------------------------
    model_path = 'blaze_face_short_range.tflite'
    if not os.path.exists(model_path):
        print(f"[*] Downloading MediaPipe face model ({model_path})...")
        url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
            out_file.write(response.read())
        print("[OK] Model downloaded.")
        
    img_path = 'friends.jpg'
    if not os.path.exists(img_path):
        print(f"[*] Downloading sample image...")
        # Using a public domain sample image
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
            out_file.write(response.read())
        print("[OK] Image downloaded.")

    # -------------------------------------------------------
    # 2. CREATE THE TASK OPTIONS
    # -------------------------------------------------------
    print("\n[*] Initializing Face Detector...")
    # BaseOptions defines which model file to use
    base_options = python.BaseOptions(model_asset_path=model_path)
    
    # FaceDetectorOptions configures how the detector behaves:
    # running_mode: IMAGE (for static files), VIDEO, or LIVE_STREAM
    # min_detection_confidence: Filter out low-probability detections
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    # -------------------------------------------------------
    # 3. RUN INFERENCE
    # -------------------------------------------------------
    # Create the face detector instance
    detector = vision.FaceDetector.create_from_options(options)

    # MediaPipe expects images in its own 'mp.Image' format.
    # We load the image directly utilizing the mp.Image API
    mp_image = mp.Image.create_from_file(img_path)

    print("[*] Running detection...")
    # Run the model
    detection_result = detector.detect(mp_image)
    
    # -------------------------------------------------------
    # 4. EXAMINE THE RESULTS
    # -------------------------------------------------------
    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")
    print(f"  Faces found: {len(detection_result.detections)}\n")
    
    for i, detection in enumerate(detection_result.detections):
        bbox = detection.bounding_box
        score = detection.categories[0].score
        print(f"  Face #{i+1} -> Confidence: {score:.2f} | BBox: [X: {bbox.origin_x}, Y: {bbox.origin_y}]")

    # -------------------------------------------------------
    # 5. VISUALIZE AND SAVE
    # -------------------------------------------------------
    # Convert mp.Image to a numpy array for OpenCV drawing
    # Note: mp.Image represents internally as RGB, so we need to switch to BGR for OpenCV
    rgb_data = mp_image.numpy_view()
    bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)

    # Draw the results!
    annotated_bgr = draw_faces_on_image(bgr_data, detection_result)

    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, '01_face_detection_result.jpg')
    cv2.imwrite(out_path, annotated_bgr)
    
    print(f"\n  Saved visualization to: {out_path}")
    print("  ✅ Script completed successfully.\n")

if __name__ == '__main__':
    main()

# Topic: Object Detection Inference with MMDetection
# Source: https://github.com/open-mmlab/mmdetection
#         https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html
# Summary:
#   1. Use the high-level DetInferencer API to perform object detection.
#   2. Load a pretrained Faster R-CNN (ResNet-50 + FPN backbone) from the model zoo.
#   3. Download a sample street image and run forward pass to detect objects.
#   4. Save visualized bounding boxes to disk and print prediction details.
#
# Key Concept:
#   MMDetection's DetInferencer wraps model loading, preprocessing, forward pass,
#   and post-processing into a single callable object — making inference as simple
#   as 3 lines of code.

from mmdet.apis import DetInferencer
import urllib.request
import os

def main():
    print("\n=======================================================")
    print("  MMDetection: Object Detection Inference")
    print("=======================================================")

    # -------------------------------------------------------
    # 1. MODEL SELECTION & INITIALIZATION
    # -------------------------------------------------------
    # We use Faster R-CNN, one of the most influential two-stage object detectors.
    # Architecture: ResNet-50 backbone + Feature Pyramid Network (FPN) neck.
    # The model name follows MMDetection's naming convention:
    #   <model>_<backbone>_<neck>_<schedule>_<dataset>
    # DetInferencer automatically downloads the config (.py) and weights (.pth)
    # from OpenMMLab's official model zoo when given just the model name.
    model_name = 'faster-rcnn_r50_fpn_1x_coco'
    print(f"\n[*] Loading model: {model_name}")
    print("    This may take a moment on first run (downloading weights ~160MB)...")

    inferencer = DetInferencer(model=model_name)
    print("[OK] Model loaded successfully!\n")

    # -------------------------------------------------------
    # 2. PREPARE A TEST IMAGE
    # -------------------------------------------------------
    # Download the official MMDetection demo image: a busy street scene
    # containing people, cars, buses, etc. — perfect for testing detection.
    img_path = 'demo.jpg'
    if not os.path.exists(img_path):
        print("[*] Downloading sample street image...")
        img_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg'
        urllib.request.urlretrieve(img_url, img_path)
        print(f"[OK] Saved to: {img_path}\n")
    else:
        print(f"[OK] Test image already exists: {img_path}\n")

    # -------------------------------------------------------
    # 3. RUN INFERENCE (FORWARD PASS)
    # -------------------------------------------------------
    # The inferencer handles everything internally:
    #   - Image loading & preprocessing (resize, normalize, pad)
    #   - GPU-accelerated forward pass through the network
    #   - Post-processing: NMS (Non-Maximum Suppression) to filter overlapping boxes
    #
    # Arguments:
    #   out_dir      : directory to save visualization + prediction JSON
    #   return_vis   : if True, include the visualized image (with bboxes) in the result dict
    #   no_save_pred : if False, also save predictions as a JSON file
    print("[*] Running object detection on the image...")
    output_dir = './outputs/01_inference'
    result = inferencer(
        img_path,
        out_dir=output_dir,
        return_vis=True,
        no_save_pred=False
    )

    # -------------------------------------------------------
    # 4. EXAMINE THE RESULTS
    # -------------------------------------------------------
    # The result dict has two keys:
    #   'predictions' : list of dicts, one per input image. Each contains:
    #       - 'labels'  : list of int class indices (COCO dataset, 80 classes)
    #       - 'scores'  : list of float confidence scores [0.0, 1.0]
    #       - 'bboxes'  : list of [x_min, y_min, x_max, y_max] coordinates
    #   'visualization' : list of numpy arrays (images with drawn bboxes)
    predictions = result['predictions'][0]
    labels = predictions['labels']
    scores = predictions['scores']
    bboxes = predictions['bboxes']

    print(f"\n{'='*55}")
    print(f"  DETECTION RESULTS")
    print(f"{'='*55}")
    print(f"  Total detections: {len(labels)}")

    # COCO class names (80 categories) — the model outputs integer indices,
    # so we map them to human-readable names for display.
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Print the top-5 most confident detections
    print(f"\n  Top-5 detections (by confidence):")
    print(f"  {'Class':<15} {'Score':>8}   {'BBox (x1, y1, x2, y2)'}")
    print(f"  {'-'*55}")
    for i in range(min(5, len(labels))):
        class_name = coco_classes[labels[i]] if labels[i] < len(coco_classes) else f"class_{labels[i]}"
        bbox_str = f"({bboxes[i][0]:.0f}, {bboxes[i][1]:.0f}, {bboxes[i][2]:.0f}, {bboxes[i][3]:.0f})"
        print(f"  {class_name:<15} {scores[i]:>8.4f}   {bbox_str}")

    print(f"\n{'='*55}")
    print(f"  OUTPUT FILES")
    print(f"{'='*55}")
    print(f"  Visualization : {output_dir}/vis/demo.jpg")
    print(f"  Predictions   : {output_dir}/preds/demo.json")
    print(f"\n  Open the visualization image to see bounding boxes drawn on the photo!")
    print("  Each box includes the class name and confidence score.\n")

if __name__ == '__main__':
    main()
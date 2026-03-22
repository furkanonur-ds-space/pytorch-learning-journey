# Topic: Understanding Detection Results & Post-Processing
# Source: https://github.com/open-mmlab/mmdetection
#         https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html
# Summary:
#   1. Deep-dive into the structure of detection result dictionaries.
#   2. Filter detections by confidence threshold and explore the effect.
#   3. Count detected objects per class and display class distribution.
#   4. Draw custom bounding boxes using OpenCV (manual visualization).
#   5. Understand Non-Maximum Suppression (NMS) and its role in post-processing.
#
# Key Concept:
#   A raw detector outputs thousands of overlapping candidate boxes. Post-processing
#   steps like confidence thresholding and NMS are essential to produce clean,
#   non-redundant detection results. Understanding these steps is crucial for
#   deploying detection models in real-world applications.

from mmdet.apis import DetInferencer
import cv2
import numpy as np
import urllib.request
import os

# COCO class names
COCO_CLASSES = [
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

# A color palette for drawing bounding boxes (BGR format for OpenCV).
# Each class gets a distinct color for better visual distinction.
np.random.seed(42)
COLORS = [(int(r), int(g), int(b)) for r, g, b in np.random.randint(50, 255, size=(80, 3))]


def main():
    print("\n=======================================================")
    print("  Understanding Detection Results & Post-Processing")
    print("=======================================================")

    # -------------------------------------------------------
    # 1. RUN DETECTION (with low threshold to get many candidates)
    # -------------------------------------------------------
    img_path = 'demo.jpg'
    if not os.path.exists(img_path):
        print("\n[*] Downloading sample image...")
        img_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg'
        urllib.request.urlretrieve(img_url, img_path)

    print("\n[*] Loading Faster R-CNN and running inference...")
    # Using pred_score_thr=0.1 to get more detections (including low-confidence ones).
    # The default threshold is 0.3, which filters out many weak detections.
    inferencer = DetInferencer(model='faster-rcnn_r50_fpn_1x_coco')
    result = inferencer(img_path, pred_score_thr=0.1)
    predictions = result['predictions'][0]

    labels = predictions['labels']
    scores = predictions['scores']
    bboxes = predictions['bboxes']

    # -------------------------------------------------------
    # 2. EXAMINE THE RAW RESULT STRUCTURE
    # -------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  RAW RESULT STRUCTURE")
    print(f"{'='*55}")
    print(f"\n  Result dict keys : {list(result.keys())}")
    print(f"  Predictions keys : {list(predictions.keys())}")
    print(f"\n  Data types:")
    print(f"    labels : {type(labels).__name__} of length {len(labels)}")
    print(f"    scores : {type(scores).__name__} of length {len(scores)}")
    print(f"    bboxes : {type(bboxes).__name__} of length {len(bboxes)}")
    if bboxes:
        print(f"\n  Single bbox format: [x_min, y_min, x_max, y_max]")
        print(f"  Example bbox     : {[f'{v:.1f}' for v in bboxes[0]]}")
        print(f"  Example label    : {labels[0]} -> '{COCO_CLASSES[labels[0]]}'")
        print(f"  Example score    : {scores[0]:.4f}")

    # -------------------------------------------------------
    # 3. CONFIDENCE THRESHOLD ANALYSIS
    # -------------------------------------------------------
    # In practice, you choose a threshold based on the precision/recall trade-off:
    #   - Higher threshold → fewer detections but more precise (fewer false positives)
    #   - Lower threshold  → more detections but noisy (more false positives)
    print(f"\n{'='*55}")
    print(f"  CONFIDENCE THRESHOLD ANALYSIS")
    print(f"{'='*55}")
    print(f"\n  How detection count changes with different thresholds:\n")
    print(f"  {'Threshold':>10}  {'Detections':>11}  {'Unique Classes':>15}")
    print(f"  {'─'*40}")

    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for thr in thresholds:
        filtered_labels = [l for l, s in zip(labels, scores) if s >= thr]
        unique_classes = len(set(filtered_labels))
        print(f"  {thr:>10.1f}  {len(filtered_labels):>11}  {unique_classes:>15}")

    # -------------------------------------------------------
    # 4. CLASS DISTRIBUTION (Object counting)
    # -------------------------------------------------------
    # Using a moderate threshold of 0.3 for meaningful detections
    thr = 0.3
    filtered = [(l, s, b) for l, s, b in zip(labels, scores, bboxes) if s >= thr]

    print(f"\n{'='*55}")
    print(f"  CLASS DISTRIBUTION (threshold={thr})")
    print(f"{'='*55}")

    class_counts = {}
    for label, score, bbox in filtered:
        name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        if name not in class_counts:
            class_counts[name] = {'count': 0, 'max_score': 0}
        class_counts[name]['count'] += 1
        class_counts[name]['max_score'] = max(class_counts[name]['max_score'], score)

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f"\n  {'Class':<18} {'Count':>6} {'Max Score':>10}  Bar")
    print(f"  {'─'*55}")
    for cls_name, info in sorted_classes:
        bar = '█' * info['count']
        print(f"  {cls_name:<18} {info['count']:>6} {info['max_score']:>10.4f}  {bar}")

    # -------------------------------------------------------
    # 5. CUSTOM BOUNDING BOX VISUALIZATION WITH OPENCV
    # -------------------------------------------------------
    # Instead of relying on MMDetection's built-in visualizer, we draw our own
    # bounding boxes. This teaches you how detection results map to pixels.
    print(f"\n{'='*55}")
    print(f"  CUSTOM BOUNDING BOX VISUALIZATION")
    print(f"{'='*55}")

    img = cv2.imread(img_path)
    img_custom = img.copy()

    for label, score, bbox in filtered:
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        color = COLORS[label % len(COLORS)]

        # Convert float coordinates to integers for drawing
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Draw the bounding box rectangle
        cv2.rectangle(img_custom, (x1, y1), (x2, y2), color, 2)

        # Prepare label text with class name and confidence score
        text = f"{class_name}: {score:.2f}"
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                      font_scale, thickness)

        # Draw a filled rectangle behind the text for readability
        cv2.rectangle(img_custom, (x1, y1 - text_h - baseline - 4),
                      (x1 + text_w, y1), color, -1)

        # Draw the text in white on top of the colored background
        cv2.putText(img_custom, text, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Save the custom visualization
    output_dir = './outputs/03_explore'
    os.makedirs(output_dir, exist_ok=True)
    custom_path = os.path.join(output_dir, 'custom_bboxes.jpg')
    cv2.imwrite(custom_path, img_custom)
    print(f"\n  Custom visualization saved to: {custom_path}")

    # -------------------------------------------------------
    # 6. NON-MAXIMUM SUPPRESSION (NMS) EXPLAINED
    # -------------------------------------------------------
    # NMS is already applied internally by the detector. Here we explain the concept
    # and show how many raw proposals exist before NMS filtering.
    print(f"\n{'='*55}")
    print(f"  NON-MAXIMUM SUPPRESSION (NMS)")
    print(f"{'='*55}")
    print("""
  What is NMS?
  ─────────────
  A detector generates many overlapping boxes for the same object.
  NMS removes redundant detections by:

    1. Sort all boxes by confidence score  (highest first)
    2. Pick the top box → keep it
    3. Compute IoU (Intersection over Union) with all remaining boxes
    4. Remove any box with IoU > threshold  (it overlaps too much → duplicate)
    5. Repeat from step 2 with the remaining boxes

  IoU Formula:
    IoU = Area_of_Intersection / Area_of_Union

  Typical NMS threshold: 0.5
    - IoU > 0.5 means the boxes overlap significantly → discard the weaker one
    - IoU < 0.5 means they are separate objects → keep both

  In MMDetection, NMS is configured in the model config file under:
    model.test_cfg.rcnn.nms  (for two-stage detectors)
    model.test_cfg.nms       (for one-stage detectors)
    """)

    # Demonstrate IoU calculation with a simple example
    def compute_iou(box_a, box_b):
        """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0

    # Find pairs of same-class boxes and compute their IoU
    if len(filtered) >= 2:
        print("  IoU examples from our detections:")
        print(f"  {'Box A (class)':<20} {'Box B (class)':<20} {'IoU':>6}")
        print(f"  {'─'*50}")
        shown = 0
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                if filtered[i][0] == filtered[j][0]:  # Same class
                    iou = compute_iou(filtered[i][2], filtered[j][2])
                    if iou > 0.01:  # Only show non-trivial overlaps
                        name = COCO_CLASSES[filtered[i][0]]
                        print(f"  {name + f' #{i}':<20} {name + f' #{j}':<20} {iou:>6.3f}")
                        shown += 1
                        if shown >= 5:
                            break
            if shown >= 5:
                break
        if shown == 0:
            print("  (No significant overlaps found — NMS already cleaned them up!)")

    print(f"\n  All outputs saved to: {output_dir}/")
    print("  ✅ Script completed successfully.\n")


if __name__ == '__main__':
    main()

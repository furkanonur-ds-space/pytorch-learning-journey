# Topic: Comparing Different Object Detection Architectures
# Source: https://github.com/open-mmlab/mmdetection
#         https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html
#         https://github.com/open-mmlab/mmdetection/blob/main/docs/en/model_zoo.md
# Summary:
#   1. Compare three fundamentally different detection paradigms side by side.
#   2. Two-Stage detector: Faster R-CNN  — proposes regions first, then classifies.
#   3. One-Stage detector: RetinaNet     — directly predicts boxes in a single pass.
#   4. Anchor-Free detector: RTMDet      — modern real-time detector without anchors.
#   5. Print and compare detection counts, speeds, and class distributions.
#
# Key Concept:
#   Object detection architectures have evolved from two-stage (accurate but slow)
#   to one-stage (fast but less accurate) to modern anchor-free designs (fast AND accurate).
#   Running all three on the same image reveals their strengths and trade-offs.

from mmdet.apis import DetInferencer
import urllib.request
import os
import time

# COCO class names (80 categories) for mapping integer labels to human-readable names.
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


def run_detector(model_name, img_path, output_subdir):
    """Load a detector, run inference, measure time, and return results."""

    print(f"\n  Loading model: {model_name}")
    inferencer = DetInferencer(model=model_name)

    # Warm-up pass — the first inference is always slower due to CUDA initialization,
    # memory allocation, and JIT compilation. We discard this result.
    print(f"  Warm-up pass...")
    _ = inferencer(img_path)

    # Timed inference pass — this gives a more realistic speed measurement.
    print(f"  Timed inference pass...")
    start_time = time.time()
    result = inferencer(img_path, out_dir=output_subdir, return_vis=True, no_save_pred=False)
    elapsed = time.time() - start_time

    predictions = result['predictions'][0]
    return predictions, elapsed


def summarize_detections(predictions, model_name, elapsed):
    """Print a summary table for one detector's results."""
    labels = predictions['labels']
    scores = predictions['scores']

    # Count how many objects of each class were detected
    class_counts = {}
    for label in labels:
        name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        class_counts[name] = class_counts.get(name, 0) + 1

    print(f"\n  Model: {model_name}")
    print(f"  {'─'*45}")
    print(f"  Total detections   : {len(labels)}")
    print(f"  Inference time     : {elapsed:.3f}s")
    if scores:
        print(f"  Avg confidence     : {sum(scores)/len(scores):.4f}")
        print(f"  Max confidence     : {max(scores):.4f}")
        print(f"  Min confidence     : {min(scores):.4f}")
    print(f"  Classes detected   : {len(class_counts)}")

    # Show the top detected classes sorted by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Class':<18} {'Count':>5}")
    print(f"  {'─'*25}")
    for cls_name, count in sorted_classes[:8]:
        print(f"  {cls_name:<18} {count:>5}")

    return class_counts


def main():
    print("\n=======================================================")
    print("  Comparing Object Detection Architectures")
    print("=======================================================")
    print("\n  We will run three different detectors on the same image")
    print("  and compare their results to understand the trade-offs.")

    # -------------------------------------------------------
    # 1. PREPARE TEST IMAGE
    # -------------------------------------------------------
    img_path = 'demo.jpg'
    if not os.path.exists(img_path):
        print("\n[*] Downloading sample image...")
        img_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg'
        urllib.request.urlretrieve(img_url, img_path)

    # -------------------------------------------------------
    # 2. DEFINE THE THREE DETECTOR ARCHITECTURES
    # -------------------------------------------------------
    # Each represents a different philosophy in object detection:
    models = {
        # TWO-STAGE DETECTOR: Region Proposal Network (RPN) first proposes ~2000
        # candidate regions, then a classifier refines each one.
        # Pros: High accuracy | Cons: Slower due to two separate stages.
        'Faster R-CNN': 'faster-rcnn_r50_fpn_1x_coco',

        # ONE-STAGE DETECTOR: Uses Focal Loss to handle the massive class imbalance
        # problem (thousands of background anchors vs. few object anchors).
        # Pros: Simpler pipeline | Cons: Can miss small objects.
        'RetinaNet': 'retinanet_r50_fpn_1x_coco',

        # ANCHOR-FREE (MODERN): RTMDet = Real-Time Models for Object Detection.
        # Eliminates handcrafted anchor boxes entirely. Uses dynamic label assignment.
        # Pros: Fast, accurate, modern | Cons: Requires more training tricks.
        'RTMDet-tiny': 'rtmdet_tiny_8xb32-300e_coco',
    }

    # -------------------------------------------------------
    # 3. RUN ALL THREE DETECTORS
    # -------------------------------------------------------
    all_results = {}
    for display_name, model_name in models.items():
        print(f"\n{'='*55}")
        print(f"  [{display_name}]")
        print(f"{'='*55}")
        output_dir = f'./outputs/02_compare/{display_name.lower().replace(" ", "_")}'
        predictions, elapsed = run_detector(model_name, img_path, output_dir)
        class_counts = summarize_detections(predictions, display_name, elapsed)
        all_results[display_name] = {
            'predictions': predictions,
            'elapsed': elapsed,
            'class_counts': class_counts
        }

    # -------------------------------------------------------
    # 4. SIDE-BY-SIDE COMPARISON TABLE
    # -------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*55}")
    print(f"\n  {'Metric':<22}", end="")
    for name in all_results:
        print(f" {name:>12}", end="")
    print()
    print(f"  {'─'*58}")

    # Total detections
    print(f"  {'Detections':<22}", end="")
    for name, data in all_results.items():
        print(f" {len(data['predictions']['labels']):>12}", end="")
    print()

    # Inference time
    print(f"  {'Time (sec)':<22}", end="")
    for name, data in all_results.items():
        print(f" {data['elapsed']:>12.3f}", end="")
    print()

    # Number of distinct classes
    print(f"  {'Unique classes':<22}", end="")
    for name, data in all_results.items():
        print(f" {len(data['class_counts']):>12}", end="")
    print()

    print(f"\n  Check the './outputs/02_compare/' folder for visualizations!")
    print("  Compare the bounding boxes drawn by each detector to see")
    print("  how they differ in coverage, precision, and speed.\n")


if __name__ == '__main__':
    main()

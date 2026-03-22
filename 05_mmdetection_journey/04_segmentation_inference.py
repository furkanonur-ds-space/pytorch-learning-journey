# Topic: Semantic Segmentation Inference with MMSegmentation
# Source: https://github.com/open-mmlab/mmsegmentation
#         https://mmsegmentation.readthedocs.io/en/latest/user_guides/3_inference.html
# Summary:
#   1. Introduction to semantic segmentation vs. object detection.
#   2. Use MMSegInferencer to perform pixel-level classification.
#   3. Load a pretrained DeepLabV3+ model (ResNet-18 backbone, Cityscapes dataset).
#   4. Run inference on a street scene and save color-coded segmentation masks.
#   5. Explore the output structure: per-pixel label maps and visualization arrays.
#
# Key Concept:
#   While object detection draws bounding BOXES around objects, semantic segmentation
#   assigns a class label to EVERY PIXEL in the image. This gives a much more detailed
#   understanding of the scene — you know exactly which pixels belong to roads, cars,
#   people, buildings, etc.

from mmseg.apis import MMSegInferencer
import urllib.request
import numpy as np
import os

def main():
    print("\n=======================================================")
    print("  MMSegmentation: Semantic Segmentation Inference")
    print("=======================================================")
    print("\n  Detection → 'There is a car at coordinates (100, 200, 300, 400)'")
    print("  Segmentation → 'Every single pixel of the car is labeled as car'\n")

    # -------------------------------------------------------
    # 1. MODEL SELECTION & INITIALIZATION
    # -------------------------------------------------------
    # DeepLabV3+ is one of the most widely-used segmentation architectures.
    # Key innovations:
    #   - Atrous (Dilated) Convolutions: enlarge the receptive field without
    #     losing spatial resolution (no pooling/striding needed).
    #   - ASPP (Atrous Spatial Pyramid Pooling): applies multiple dilated convolutions
    #     with different rates to capture multi-scale context.
    #   - Encoder-Decoder structure: the '+' in DeepLabV3+ adds a decoder module
    #     that sharpens object boundaries.
    #
    # We use a lightweight ResNet-18 backbone trained on the Cityscapes dataset,
    # which is designed for urban street scene understanding.
    model_name = 'deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024'
    print(f"[*] Loading model: {model_name}")
    print("    Architecture : DeepLabV3+ with ResNet-18 backbone")
    print("    Dataset      : Cityscapes (19 urban classes)")
    print("    This may take a moment on first run (downloading weights)...")

    inferencer = MMSegInferencer(model=model_name)
    print("[OK] Model loaded successfully!\n")

    # -------------------------------------------------------
    # 2. PREPARE A TEST IMAGE
    # -------------------------------------------------------
    # Download a sample image appropriate for the Cityscapes-trained model.
    # Cityscapes models are trained on European street scenes, so they work best
    # with similar urban driving imagery.
    img_path = 'seg_demo.jpg'
    if not os.path.exists(img_path):
        print("[*] Downloading sample image for segmentation...")
        # Using MMDetection's demo image — a street scene with vehicles and pedestrians
        img_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg'
        urllib.request.urlretrieve(img_url, img_path)
        print(f"[OK] Saved to: {img_path}\n")
    else:
        print(f"[OK] Test image already exists: {img_path}\n")

    # -------------------------------------------------------
    # 3. RUN SEGMENTATION INFERENCE
    # -------------------------------------------------------
    # Unlike detection (which outputs boxes), segmentation outputs:
    #   - A prediction mask: H x W array where each pixel value is a class index
    #   - A visualization: H x W x 3 color image with each class in a unique color
    #
    # Arguments:
    #   out_dir      : root directory for saving outputs
    #   img_out_dir  : subdirectory name for visualization images
    #   pred_out_dir : subdirectory name for raw prediction masks
    #   opacity      : transparency of the segmentation overlay (0=transparent, 1=opaque)
    print("[*] Running semantic segmentation...")
    output_dir = './outputs/04_segmentation'
    result = inferencer(
        img_path,
        out_dir=output_dir,
        img_out_dir='vis',
        pred_out_dir='pred',
        opacity=0.7,
        return_vis=True
    )

    # -------------------------------------------------------
    # 4. EXAMINE OUTPUT STRUCTURE
    # -------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  SEGMENTATION OUTPUT STRUCTURE")
    print(f"{'='*55}")
    print(f"\n  Result dict keys: {list(result.keys())}")

    # The 'predictions' value is a numpy array with shape (H, W)
    # Each pixel value is an integer class index [0, num_classes-1]
    pred_mask = result['predictions'][0]
    if isinstance(pred_mask, np.ndarray):
        print(f"\n  Prediction mask shape : {pred_mask.shape}  (Height x Width)")
        print(f"  Prediction mask dtype : {pred_mask.dtype}")
        print(f"  Unique class indices  : {np.unique(pred_mask).tolist()}")
    else:
        # Handle case where predictions is a SegDataSample
        print(f"\n  Prediction type: {type(pred_mask)}")
        pred_mask = np.array(pred_mask)
        if pred_mask.ndim >= 2:
            print(f"  Prediction mask shape : {pred_mask.shape}")

    # The 'visualization' value is a numpy array with shape (H, W, 3)
    # It's the original image blended with color-coded segmentation mask
    vis = result['visualization'][0]
    if isinstance(vis, np.ndarray):
        print(f"\n  Visualization shape   : {vis.shape}  (H x W x RGB)")
        print(f"  Visualization dtype   : {vis.dtype}")

    # -------------------------------------------------------
    # 5. CITYSCAPES CLASS REFERENCE
    # -------------------------------------------------------
    # Cityscapes has 19 semantic classes for urban scene understanding.
    # This is the mapping from class index to class name.
    cityscapes_classes = [
        'road', 'sidewalk', 'building', 'wall', 'fence',           # 0-4
        'pole', 'traffic light', 'traffic sign', 'vegetation',     # 5-8
        'terrain', 'sky', 'person', 'rider', 'car',                # 9-13
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'           # 14-18
    ]

    print(f"\n{'='*55}")
    print(f"  CITYSCAPES CLASSES (19 categories)")
    print(f"{'='*55}\n")
    for i, cls_name in enumerate(cityscapes_classes):
        print(f"  [{i:>2}] {cls_name}")

    # -------------------------------------------------------
    # 6. PER-CLASS PIXEL STATISTICS
    # -------------------------------------------------------
    if isinstance(pred_mask, np.ndarray) and pred_mask.ndim == 2:
        total_pixels = pred_mask.size

        print(f"\n{'='*55}")
        print(f"  PER-CLASS PIXEL COVERAGE")
        print(f"{'='*55}")
        print(f"\n  Total pixels: {total_pixels:,}")
        print(f"\n  {'Class':<16} {'Pixels':>10} {'Coverage':>10}")
        print(f"  {'─'*40}")

        unique_labels, counts = np.unique(pred_mask, return_counts=True)
        # Sort by pixel count in descending order
        sorted_indices = np.argsort(-counts)
        for idx in sorted_indices:
            label = unique_labels[idx]
            count = counts[idx]
            pct = count / total_pixels * 100
            name = cityscapes_classes[label] if label < len(cityscapes_classes) else f"class_{label}"
            bar = '█' * int(pct / 2)
            print(f"  {name:<16} {count:>10,} {pct:>9.1f}%  {bar}")

    # -------------------------------------------------------
    # 7. DETECTION vs. SEGMENTATION SUMMARY
    # -------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  DETECTION vs. SEGMENTATION")
    print(f"{'='*55}")
    print("""
  ┌──────────────────┬─────────────────────────────────────┐
  │                  │                                     │
  │  Detection       │  Bounding boxes around objects      │
  │  (MMDetection)   │  Output: list of (class, bbox)      │
  │                  │  Use case: counting, tracking       │
  │                  │                                     │
  ├──────────────────┼─────────────────────────────────────┤
  │                  │                                     │
  │  Segmentation    │  Per-pixel class labels             │
  │  (MMSegmentation)│  Output: H x W label map            │
  │                  │  Use case: autonomous driving,      │
  │                  │           medical imaging           │
  │                  │                                     │
  └──────────────────┴─────────────────────────────────────┘
    """)
    print(f"  Output files saved to: {output_dir}/")
    print(f"  Open '{output_dir}/vis/' to see the color-coded segmentation overlay!")
    print("  ✅ Segmentation inference completed successfully.\n")


if __name__ == '__main__':
    main()

# Topic: Analyzing Segmentation Outputs & Pixel-Level Predictions
# Source: https://github.com/open-mmlab/mmsegmentation
#         https://mmsegmentation.readthedocs.io/en/latest/user_guides/3_inference.html
# Summary:
#   1. Run segmentation with two different architectures (DeepLabV3+ vs PSPNet).
#   2. Extract and analyze per-pixel label maps programmatically.
#   3. Create custom colored overlays using NumPy and OpenCV.
#   4. Compare how different models segment the same image.
#   5. Compute per-class IoU (Intersection over Union) between the two models.
#
# Key Concept:
#   Different segmentation architectures can produce different results on the same
#   image. Comparing them reveals which areas are "easy" to segment (both agree)
#   and which are "hard" (they disagree). This analysis helps you choose the right
#   model for your specific use case.

from mmseg.apis import MMSegInferencer
import cv2
import numpy as np
import urllib.request
import os

# Cityscapes 19 classes with carefully chosen RGB colors for visualization.
# These colors follow the official Cityscapes color convention.
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

CITYSCAPES_PALETTE = [
    (128, 64, 128),   # road        - purple
    (244, 35, 232),   # sidewalk    - pink
    (70, 70, 70),     # building    - dark gray
    (102, 102, 156),  # wall        - blue-gray
    (190, 153, 153),  # fence       - light gray
    (153, 153, 153),  # pole        - gray
    (250, 170, 30),   # traffic light - orange
    (220, 220, 0),    # traffic sign  - yellow
    (107, 142, 35),   # vegetation  - olive green
    (152, 251, 152),  # terrain     - light green
    (70, 130, 180),   # sky         - steel blue
    (220, 20, 60),    # person      - crimson
    (255, 0, 0),      # rider       - red
    (0, 0, 142),      # car         - dark blue
    (0, 0, 70),       # truck       - navy
    (0, 60, 100),     # bus         - dark teal
    (0, 80, 100),     # train       - teal
    (0, 0, 230),      # motorcycle  - blue
    (119, 11, 32),    # bicycle     - dark red
]


def create_colored_mask(pred_mask, palette):
    """Convert a label map (H, W) into a colored RGB image (H, W, 3).

    Args:
        pred_mask: numpy array of shape (H, W) with integer class indices.
        palette: list of (R, G, B) tuples, one per class.

    Returns:
        colored: numpy array of shape (H, W, 3) with RGB values.
    """
    h, w = pred_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        # Create a boolean mask for all pixels of this class
        mask = pred_mask == class_idx
        colored[mask] = color
    return colored


def compute_per_class_iou(mask_a, mask_b, num_classes):
    """Compute IoU between two segmentation masks for each class.

    IoU (Intersection over Union) for segmentation is computed per-class:
        For each class c:
            intersection = pixels where both masks predict class c
            union = pixels where either mask predicts class c
            IoU(c) = intersection / union
    """
    ious = {}
    for c in range(num_classes):
        pred_a = (mask_a == c)
        pred_b = (mask_b == c)
        intersection = np.logical_and(pred_a, pred_b).sum()
        union = np.logical_or(pred_a, pred_b).sum()
        if union > 0:
            ious[c] = intersection / union
    return ious


def main():
    print("\n=======================================================")
    print("  Analyzing Segmentation: DeepLabV3+ vs PSPNet")
    print("=======================================================")

    # -------------------------------------------------------
    # 1. PREPARE TEST IMAGE
    # -------------------------------------------------------
    img_path = 'seg_demo.jpg'
    if not os.path.exists(img_path):
        print("\n[*] Downloading sample image...")
        img_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg'
        urllib.request.urlretrieve(img_url, img_path)

    output_dir = './outputs/05_seg_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------
    # 2. DEFINE THE TWO SEGMENTATION ARCHITECTURES
    # -------------------------------------------------------
    models = {
        # DEEPLABV3+ (Chen et al., 2018)
        # Key idea: Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale
        # context using dilated convolutions at different rates.
        # The '+' adds an encoder-decoder structure for sharper boundaries.
        'DeepLabV3+': 'deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024',

        # PSPNET - Pyramid Scene Parsing Network (Zhao et al., 2017)
        # Key idea: Pyramid Pooling Module that pools features at 4 different
        # scales (1x1, 2x2, 3x3, 6x6) and concatenates them. This helps the
        # model understand both local details and global context.
        'PSPNet': 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024',
    }

    # -------------------------------------------------------
    # 3. RUN BOTH MODELS
    # -------------------------------------------------------
    results = {}
    for display_name, model_name in models.items():
        print(f"\n  Loading {display_name}: {model_name}")
        inferencer = MMSegInferencer(model=model_name)
        print(f"  Running inference...")
        result = inferencer(img_path)
        pred_mask = result['predictions'][0]
        if not isinstance(pred_mask, np.ndarray):
            pred_mask = np.array(pred_mask)
        results[display_name] = pred_mask
        print(f"  [OK] {display_name} complete — mask shape: {pred_mask.shape}")

    # -------------------------------------------------------
    # 4. PER-CLASS PIXEL COVERAGE COMPARISON
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PER-CLASS PIXEL COVERAGE COMPARISON")
    print(f"{'='*60}")

    # Get pixel counts for each model
    model_names = list(results.keys())
    mask_a = results[model_names[0]]
    mask_b = results[model_names[1]]
    total_pixels = mask_a.size

    print(f"\n  {'Class':<16}", end="")
    for name in model_names:
        print(f" {name + ' %':>14}", end="")
    print(f" {'Difference':>12}")
    print(f"  {'─'*58}")

    all_labels = set(np.unique(mask_a).tolist()) | set(np.unique(mask_b).tolist())
    rows = []
    for label in sorted(all_labels):
        name = CITYSCAPES_CLASSES[label] if label < len(CITYSCAPES_CLASSES) else f"cls_{label}"
        pct_a = np.sum(mask_a == label) / total_pixels * 100
        pct_b = np.sum(mask_b == label) / total_pixels * 100
        diff = abs(pct_a - pct_b)
        rows.append((name, pct_a, pct_b, diff))

    # Sort by maximum coverage across both models
    rows.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    for name, pct_a, pct_b, diff in rows:
        diff_marker = ' ⚠️' if diff > 5.0 else ''
        print(f"  {name:<16} {pct_a:>13.1f}% {pct_b:>13.1f}% {diff:>11.1f}%{diff_marker}")

    # -------------------------------------------------------
    # 5. AGREEMENT/DISAGREEMENT ANALYSIS
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  MODEL AGREEMENT ANALYSIS")
    print(f"{'='*60}")

    agreement_mask = (mask_a == mask_b)
    agreement_pct = np.sum(agreement_mask) / total_pixels * 100
    disagreement_pct = 100.0 - agreement_pct

    print(f"\n  Pixels where both models agree   : {agreement_pct:.1f}%")
    print(f"  Pixels where models disagree     : {disagreement_pct:.1f}%")
    print(f"\n  Higher agreement = more 'confident' segmentation")
    print(f"  Disagreement areas = ambiguous regions (e.g., object boundaries)")

    # -------------------------------------------------------
    # 6. PER-CLASS IoU BETWEEN THE TWO MODELS
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PER-CLASS IoU (Model Agreement per Class)")
    print(f"{'='*60}")
    print(f"\n  IoU = Intersection / Union")
    print(f"  IoU = 1.0 means both models perfectly agree on that class")
    print(f"  IoU = 0.0 means complete disagreement\n")

    ious = compute_per_class_iou(mask_a, mask_b, num_classes=19)

    print(f"  {'Class':<16} {'IoU':>8}  {'Agreement'}")
    print(f"  {'─'*50}")
    for class_idx in sorted(ious.keys(), key=lambda x: ious[x], reverse=True):
        iou = ious[class_idx]
        name = CITYSCAPES_CLASSES[class_idx] if class_idx < len(CITYSCAPES_CLASSES) else f"cls_{class_idx}"
        bar = '█' * int(iou * 20)
        print(f"  {name:<16} {iou:>8.3f}  {bar}")

    if ious:
        mean_iou = sum(ious.values()) / len(ious)
        print(f"\n  Mean IoU across classes: {mean_iou:.3f}")

    # -------------------------------------------------------
    # 7. CUSTOM VISUALIZATION WITH OPENCV
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  CUSTOM VISUALIZATIONS")
    print(f"{'='*60}")

    img = cv2.imread(img_path)

    # Create color-coded masks for each model
    for model_name, pred_mask in results.items():
        # Convert prediction mask to colored image
        colored = create_colored_mask(pred_mask, CITYSCAPES_PALETTE)
        # OpenCV uses BGR, but our palette is RGB — convert
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

        # Resize colored mask to match original image size (models may resize internally)
        if colored_bgr.shape[:2] != img.shape[:2]:
            colored_bgr = cv2.resize(colored_bgr, (img.shape[1], img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # Blend original image with segmentation mask (alpha blending)
        alpha = 0.6  # Opacity of the segmentation overlay
        blended = cv2.addWeighted(img, 1 - alpha, colored_bgr, alpha, 0)

        # Save individual model outputs
        safe_name = model_name.lower().replace("+", "plus").replace(" ", "_")
        mask_path = os.path.join(output_dir, f'{safe_name}_mask.jpg')
        blend_path = os.path.join(output_dir, f'{safe_name}_overlay.jpg')
        cv2.imwrite(mask_path, colored_bgr)
        cv2.imwrite(blend_path, blended)
        print(f"\n  [{model_name}]")
        print(f"    Raw mask  : {mask_path}")
        print(f"    Overlay   : {blend_path}")

    # Create a disagreement heatmap — white where models agree, red where they disagree
    disagree_vis = np.full_like(img, 255)  # Start with white
    disagree_pixels = ~agreement_mask
    if disagree_vis.shape[:2] != mask_a.shape:
        # Resize disagreement mask to match image
        disagree_pixels_resized = cv2.resize(disagree_pixels.astype(np.uint8),
                                              (img.shape[1], img.shape[0]),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        disagree_pixels_resized = disagree_pixels

    disagree_vis[disagree_pixels_resized] = [0, 0, 255]  # Red for disagreement
    disagree_path = os.path.join(output_dir, 'disagreement_map.jpg')
    cv2.imwrite(disagree_path, disagree_vis)
    print(f"\n  [Disagreement Map]")
    print(f"    White = agreement, Red = disagreement")
    print(f"    Saved to: {disagree_path}")

    # -------------------------------------------------------
    # 8. ARCHITECTURE COMPARISON SUMMARY
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ARCHITECTURE COMPARISON")
    print(f"{'='*60}")
    print("""
  ┌──────────────────┬──────────────────────────────────────┐
  │  DeepLabV3+      │  Uses Atrous Spatial Pyramid Pooling │
  │  (Chen 2018)     │  + Encoder-Decoder for sharp edges   │
  │                  │  Best at: fine boundary details      │
  ├──────────────────┼──────────────────────────────────────┤
  │  PSPNet          │  Uses Pyramid Pooling Module at      │
  │  (Zhao 2017)     │  multiple scales (1x1 to 6x6)       │
  │                  │  Best at: global scene understanding │
  └──────────────────┴──────────────────────────────────────┘
    """)
    print(f"  All outputs saved to: {output_dir}/")
    print("  ✅ Segmentation analysis completed successfully.\n")


if __name__ == '__main__':
    main()

"""
Verification script to ensure YOLOv8 recognizes all 80 COCO classes.

This script loads the model and prints all available classes to verify
the system can detect all objects from the COCO dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .detector import YoloDetector


# Official COCO class list (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def verify_classes(model_path: str = "models/yolov8n.pt") -> None:
    """
    Load the detector and verify it has all 80 COCO classes.
    """
    print("Loading YOLOv8 model...")
    detector = YoloDetector(model_path=model_path, conf_threshold=0.4)
    
    print(f"\n✓ Model loaded successfully!")
    print(f"✓ Number of classes in model: {detector.get_class_count()}")
    
    # Get all classes from the model
    model_classes = detector.get_all_classes()
    
    print(f"\n{'='*60}")
    print("VERIFICATION REPORT")
    print(f"{'='*60}\n")
    
    # Check if we have exactly 80 classes
    if len(model_classes) == 80:
        print("✓ Model has 80 classes (correct for COCO dataset)")
    else:
        print(f"⚠ Warning: Model has {len(model_classes)} classes, expected 80")
    
    print(f"\nAll classes the model can detect:")
    print(f"{'-'*60}")
    
    # Print classes in organized format
    for class_id in sorted(model_classes.keys()):
        class_name = model_classes[class_id]
        in_coco = "✓" if class_name in COCO_CLASSES else "✗"
        print(f"  [{class_id:2d}] {in_coco} {class_name}")
    
    # Verify all COCO classes are present
    print(f"\n{'='*60}")
    print("COCO CLASS COVERAGE CHECK")
    print(f"{'='*60}\n")
    
    missing_classes = []
    for coco_class in COCO_CLASSES:
        if coco_class not in model_classes.values():
            missing_classes.append(coco_class)
    
    if not missing_classes:
        print("✓ All 80 COCO classes are available in the model!")
    else:
        print(f"✗ Missing {len(missing_classes)} COCO classes:")
        for cls in missing_classes:
            print(f"    - {cls}")
    
    # Check for extra classes not in COCO
    model_class_names = set(model_classes.values())
    coco_class_set = set(COCO_CLASSES)
    extra_classes = model_class_names - coco_class_set
    
    if extra_classes:
        print(f"\n⚠ Model has {len(extra_classes)} extra classes not in COCO:")
        for cls in sorted(extra_classes):
            print(f"    + {cls}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total classes in model: {len(model_classes)}")
    print(f"COCO classes covered: {len(COCO_CLASSES) - len(missing_classes)}/{len(COCO_CLASSES)}")
    print(f"Status: {'✓ VERIFIED - All COCO classes available' if not missing_classes else '✗ INCOMPLETE'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that YOLOv8 model recognizes all 80 COCO classes."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.pt",
        help="Path to YOLOv8 model weights.",
    )
    args = parser.parse_args()
    verify_classes(model_path=args.model)


if __name__ == "__main__":
    main()


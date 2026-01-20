## Real-Time Scene Description System using Computer Vision

Real-Time Scene Description is a Python-based system that processes live video (webcam or file), detects objects with YOLOv8, infers simple human–object interactions, and generates concise natural-language captions such as:

- **"A person holding a bottle"**
- **"A person using a mobile phone"**

The pipeline is designed to be robust in noisy and low-light environments via classical computer-vision preprocessing and to produce a captioned video recording of the session.

---

### Features

- **Real-time video processing**: Works with a webcam or video file input.
- **YOLOv8 object detection**: Uses the lightweight `yolov8n` model for efficient inference.
- **Human–object interaction inference**: Identifies persons and overlapping objects, classifying interactions such as **holding** or **using**.
- **Scene caption generation**: Converts structured interactions into short, human-readable descriptions.
- **Robust preprocessing**: Noise reduction (fastNlMeansDenoisingColored) and contrast enhancement (CLAHE in LAB space) for challenging scenes.
- **Captioned video export**: Saves the annotated output video to `data/outputs/output.mp4`.
- **Full COCO class support**: Recognizes all **80 object classes** from the COCO dataset, including people, vehicles, animals, electronics, furniture, food, and everyday objects.

---

### Tech Stack

- **Language**: Python 3.9+
- **Deep Learning / Detection**: Ultralytics YOLOv8 (`yolov8n.pt`), PyTorch, TorchVision
- **Computer Vision**: OpenCV
- **Numerical Computing**: NumPy

---

### Project Structure

```text
real-time-scene-description/
│
├── data/
│   └── outputs/
│       └── output.mp4          # Generated captioned video (created at runtime)
│
├── models/
│   └── yolov8n.pt              # YOLOv8n model weights (user-provided)
│
├── src/
│   ├── preprocess.py           # Denoising and contrast enhancement
│   ├── detector.py             # YOLOv8-based object detection
│   ├── interaction.py          # Human–object interaction inference
│   ├── captioner.py            # Natural-language caption generation
│   ├── video_utils.py          # Video I/O, resizing, overlay utilities
│   └── main.py                 # End-to-end pipeline entry point
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

### Architecture Overview

- **Preprocessing (`preprocess.py`)**
  - Applies **fast non-local means denoising** (`fastNlMeansDenoisingColored`) to suppress sensor noise, particularly in low light.
  - Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) in **LAB** color space to locally enhance contrast while avoiding over-amplification of noise.
  - Exposes a single entry point: `enhance_frame(frame)` → enhanced BGR frame.

- **Object Detection (`detector.py`)**
  - Wraps Ultralytics **YOLOv8n** in a `YoloDetector` class.
  - Loads the model **once** at startup from `models/yolov8n.pt`.
  - Trained on **COCO dataset** with **80 object classes** including:
    - **People**: person
    - **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
    - **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    - **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, etc.
    - **Furniture**: chair, couch, bed, dining table, toilet
    - **Food & drinks**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, pizza, etc.
    - **Sports equipment**: sports ball, kite, baseball bat, skateboard, surfboard, tennis racket
    - **Personal items**: backpack, umbrella, handbag, tie, suitcase, etc.
    - **And 40+ more classes** (see verification script below)
  - For each frame, returns a list of detections as:
    - **`(label, confidence, (x1, y1, x2, y2))`**, where coordinates are pixel-space bounding boxes.
  - Filters detections with a confidence threshold **≥ 0.4**.
  - Verify all 80 classes are available: `python -m src.verify_classes`

- **Human–Object Interaction Inference (`interaction.py`)**
  - Splits detections into **persons** and **non-person objects** (labels other than `"person"`).
  - Computes **IoU (Intersection-over-Union)** between person and object bounding boxes.
  - If IoU exceeds a small threshold (default `0.1`), the pair is treated as an interaction.
  - Uses heuristics on object labels (e.g., `"cell phone"`, `"laptop"`, `"keyboard"`) to classify the interaction as:
    - **"using"** for device-like objects.
    - **"holding"** otherwise.
  - Returns a structured list of `Interaction` dataclass instances.

- **Caption Generation (`captioner.py`)**
  - Converts the list of `Interaction` instances into concise English phrases:
    - Example: `"a person holding a bottle"`, `"a person using a mobile phone"`.
  - Aggregates multiple interactions and produces a single caption string:
    - Example: `"A person holding a bottle and a person using a mobile phone"`.
  - Provides a fallback caption when no interactions are detected:
    - `"A scene with no clear human–object interaction"`.

- **Video Utilities (`video_utils.py`)**
  - Opens a video source (`webcam` index or file path) using OpenCV.
  - Normalizes frame size (default **640×480**) using `resize_frame`.
  - Creates a video writer for MP4 output using `mp4v` codec.
  - Overlays captions on frames with a readable text box using `overlay_caption`.

- **Main Pipeline (`main.py`)**
  - Accepts CLI arguments for:
    - `--source`: webcam index (e.g. `"0"`) or video file path.
    - `--output`: path for the captioned output video (default `data/outputs/output.mp4`).
    - `--width`, `--height`: output frame size.
    - `--fps`: target frame rate for output video (default `30.0` fps).
    - `--skip-frames`: process every Nth frame for detection (default `2`, higher = faster).
    - `--no-preprocess`: skip expensive preprocessing for faster performance.
    - `--device`: YOLO device (e.g. `"cpu"`, `"cuda"`).
    - `--model`: model weights path (default `models/yolov8n.pt`).
  - For each frame:
    1. Resizes frame to target size.
    2. Enhances the frame using `enhance_frame`.
    3. Runs YOLO detection with `YoloDetector`.
    4. Infers interactions via `infer_interactions`.
    5. Generates a caption via `generate_caption`.
    6. Overlays the caption and bounding boxes for visualization.
    7. Displays the annotated frame live and writes it to `output.mp4`.
  - Exits cleanly when **`q`** is pressed or the stream ends.

---

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>.git
   cd real-time-scene-description
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download YOLOv8n model weights**

   Download `yolov8n.pt` (the nano YOLOv8 model) from the official Ultralytics model zoo and place it in the `models/` directory:

   ```text
   real-time-scene-description/
   ├── models/
   │   └── yolov8n.pt
   ```

   You can also obtain it programmatically (e.g., via `from ultralytics import YOLO; YOLO("yolov8n.pt")`), but for reproducibility we assume it is present on disk.

---

### How to Run

All commands below are assumed to be executed from the project root (`real-time-scene-description/`).

- **Webcam input (default)**

  ```bash
  python -m src.main
  ```

  This uses webcam index `0`, outputs to `data/outputs/output.mp4`, and processes frames at `640x480`.

- **Explicit webcam index**

  ```bash
  python -m src.main --source 1
  ```

  Uses webcam index `1` instead of `0`.

- **Video file input**

  ```bash
  python -m src.main --source path/to/video.mp4
  ```

- **Specify output path, resolution, and frame rate**

  ```bash
  python -m src.main \
      --source 0 \
      --output data/outputs/output.mp4 \
      --width 1280 \
      --height 720 \
      --fps 30
  ```

- **Optimize for performance (skip preprocessing, higher frame skip)**

  ```bash
  python -m src.main --no-preprocess --skip-frames 3 --fps 30
  ```

- **Select computation device (if you have a GPU)**

  ```bash
  python -m src.main --device cuda
  ```

During execution, a window titled **"Real-Time Scene Description (press 'q' to quit)"** will display the live annotated frames. Press **`q`** to terminate and finalize the output video.

- **Verify object recognition capabilities**

  ```bash
  python -m src.verify_classes
  ```

  This will load the model and print all 80 COCO classes the system can detect, confirming complete coverage of the COCO dataset.

---

### Example Captions

Depending on the scene and detections, the system may produce captions such as:

- **"A person holding a bottle"**
- **"A person using a mobile phone"**
- **"A person using a laptop and a person holding a cup"**
- **"A scene with no clear human–object interaction"**

These captions are generated via simple, interpretable heuristics on bounding-box overlaps and object labels, making the system predictable and easy to extend.

---

### Applications

- **Assistive technology**: Provide coarse real-time scene descriptions for users with visual impairments.
- **Surveillance and monitoring**: Quickly summarize activities in a scene for human operators.
- **Human–computer interaction research**: Prototype interaction-aware interfaces using off-the-shelf object detectors.
- **Dataset exploration**: Rapidly inspect video datasets with lightweight, textual scene summaries.

---

### Future Improvements

- **Richer HOI modeling**: Replace heuristic interaction rules with a dedicated human–object interaction model or transformer-based captioner.
- **Temporal reasoning**: Incorporate motion and temporal context (e.g., tracking, action recognition) for more accurate descriptions.
- **Language generation**: Use large language models to refine and diversify captions while maintaining correctness.
- **Uncertainty modeling**: Expose detection and interaction confidence scores directly in the caption text.
- **Multi-person and multi-camera scenarios**: Scale the architecture to handle complex scenes with many agents and inputs.

---

### License

This project is intended for research and educational use. Please ensure that your use of YOLOv8 and associated models complies with the Ultralytics license and any third-party dependencies.



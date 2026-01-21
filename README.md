# Real-Time Scene Description System

A production-ready Python system that processes live video (webcam or file), detects objects using YOLOv8, infers human–object interactions, and generates concise natural-language scene descriptions in real-time.

## 🎯 Overview

This system combines state-of-the-art object detection with simple interaction reasoning to produce real-time scene captions. Perfect for assistive technology, surveillance monitoring, or human–computer interaction research.

### Key Capabilities

- **Real-time object detection** using YOLOv8n (recognizes 80 COCO classes)
- **Human–object interaction inference** with bounding box overlap analysis
- **Natural language caption generation** (e.g., "A person holding a bottle")
- **Robust preprocessing** for low-light and noisy environments
- **Performance optimized** with configurable frame skipping and preprocessing options
- **Automatic model download** - works out of the box after cloning

### Example Output

```
"A person using a mobile phone"
"A person holding a bottle and a person using a laptop"
"A scene with no clear human–object interaction"
```

---

## ✨ Features

### Core Features
- **Real-time video processing**: Works seamlessly with webcam or video file input
- **80 COCO object classes**: Detects people, vehicles, animals, electronics, furniture, food, and more
- **Human–object interaction detection**: Identifies when persons interact with objects (holding/using)
- **Scene caption generation**: Converts detections into readable English descriptions
- **Live visualization**: Real-time display with bounding boxes and captions
- **Video export**: Saves captioned output video to MP4 format

### Performance Features
- **Frame skipping**: Process every Nth frame for higher FPS (default: every 5th frame)
- **Configurable detection size**: Smaller input = faster inference (default: 320×320)
- **Optional preprocessing**: Skip expensive operations for maximum speed
- **FPS monitoring**: Real-time FPS counter displayed on video
- **GPU support**: Automatic CUDA detection when available

### Robustness Features
- **Noise reduction**: Fast non-local means denoising for low-light scenes
- **Contrast enhancement**: CLAHE in LAB color space for challenging lighting
- **Configurable confidence thresholds**: Balance accuracy vs. speed

---

## 🛠️ Tech Stack

- **Python 3.9+**: Core language
- **Ultralytics YOLOv8**: Object detection engine (`yolov8n.pt` model)
- **PyTorch & TorchVision**: Deep learning backend
- **OpenCV**: Video processing, preprocessing, and visualization
- **NumPy**: Numerical computations

---

## 📁 Project Structure

```
object_detector/
│
├── data/
│   └── outputs/
│       └── .gitkeep              # Output videos saved here (auto-created)
│
├── models/
│   └── .gitkeep                  # Model weights directory (auto-downloaded)
│
├── src/
│   ├── __init__.py               # Package initialization
│   ├── preprocess.py             # Frame enhancement (denoising, CLAHE)
│   ├── detector.py               # YOLOv8 object detection wrapper
│   ├── interaction.py            # Human–object interaction inference
│   ├── captioner.py              # Natural language caption generation
│   ├── video_utils.py            # Video I/O and overlay utilities
│   ├── main.py                   # Main pipeline entry point
│   └── verify_classes.py         # Utility to verify 80 COCO classes
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🏗️ Architecture

### Pipeline Overview

```
Video Input → Preprocessing → Object Detection → Interaction Inference → Caption Generation → Output
```

### Module Details

#### 1. Preprocessing (`preprocess.py`)
- **Fast Non-Local Means Denoising**: Reduces sensor noise, especially in low-light conditions
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization in LAB color space
- **Configurable**: Can be disabled for maximum performance (`--no-preprocess`)

#### 2. Object Detection (`detector.py`)
- **YOLOv8n Model**: Lightweight nano variant for real-time performance
- **80 COCO Classes**: Full COCO dataset coverage (see complete list below)
- **Efficient Inference**: Supports configurable input size (default: 320×320)
- **Auto-Download**: Automatically fetches model weights if missing

#### 3. Interaction Inference (`interaction.py`)
- **Bounding Box Overlap**: Uses IoU (Intersection-over-Union) to detect interactions
- **Person Detection**: Separates persons from other objects
- **Interaction Classification**:
  - **"using"**: For device-like objects (phone, laptop, keyboard, etc.)
  - **"holding"**: For other objects (bottle, cup, etc.)
- **Threshold**: Default IoU threshold of 0.1

#### 4. Caption Generation (`captioner.py`)
- **Natural Language**: Converts structured interactions to readable text
- **Multiple Interactions**: Handles multiple person–object pairs gracefully
- **Fallback**: Provides default caption when no interactions detected
- **Grammatical Correctness**: Proper article usage (a/an)

#### 5. Video Utilities (`video_utils.py`)
- **Input Handling**: Supports webcam indices and file paths
- **Frame Resizing**: Consistent output dimensions
- **Overlay Rendering**: Text boxes with proper contrast for readability
- **MP4 Export**: Uses `mp4v` codec for compatibility

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Webcam (for live processing) or video files
- ~500 MB disk space (for dependencies and model)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nayan2723/object_detector.git
   cd object_detector
   ```

2. **Create virtual environment** (highly recommended)
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

4. **Verify installation**
   ```bash
   python -m src.verify_classes
   ```
   This will auto-download the YOLOv8 model (~6MB) and verify all 80 COCO classes are available.

**That's it!** The system is ready to use. Model weights are automatically downloaded on first run.

---

## 💻 Usage

### Basic Usage

**Webcam (default)**
```bash
python -m src.main
```

**Video file**
```bash
python -m src.main --source path/to/video.mp4
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `"0"` | Video source: webcam index (e.g., `"0"`, `"1"`) or video file path |
| `--output` | `"data/outputs/output.mp4"` | Path for output video file |
| `--width` | `320` | Output frame width (lower = faster) |
| `--height` | `240` | Output frame height (lower = faster) |
| `--fps` | `30.0` | Target frame rate for output video |
| `--skip-frames` | `5` | Process every Nth frame (higher = faster, less accurate) |
| `--no-preprocess` | `True` | Skip preprocessing for maximum speed |
| `--preprocess` | - | Enable preprocessing (CLAHE) - slower but better for low-light |
| `--conf-threshold` | `0.3` | YOLO confidence threshold (lower = faster, more detections) |
| `--detection-size` | `320` | YOLO input size in pixels (smaller = faster: 224, 320, 416, 640) |
| `--device` | `None` | Computation device: `"cpu"` or `"cuda"` (auto-detects if None) |
| `--model` | `"models/yolov8n.pt"` | Path to YOLOv8 model weights |

### Common Usage Patterns

#### Maximum Performance (Fastest FPS)
```bash
python -m src.main \
    --no-preprocess \
    --skip-frames 10 \
    --detection-size 224 \
    --width 240 \
    --height 180 \
    --conf-threshold 0.25
```

#### High Quality (Better Accuracy)
```bash
python -m src.main \
    --preprocess \
    --skip-frames 2 \
    --detection-size 640 \
    --width 640 \
    --height 480 \
    --conf-threshold 0.4
```

#### GPU Acceleration (if available)
```bash
python -m src.main --device cuda
```

#### Custom Output Location
```bash
python -m src.main \
    --source 0 \
    --output my_output_video.mp4 \
    --width 1280 \
    --height 720 \
    --fps 30
```

### During Execution

- **Live Window**: Real-time video feed with bounding boxes and captions
- **FPS Display**: Bottom-left shows current FPS and target FPS
- **Exit**: Press `q` to quit and save output video
- **Output**: Video saved to specified path (default: `data/outputs/output.mp4`)

---

## 📊 Performance Optimization

### Understanding Performance Bottlenecks

1. **YOLO Inference**: Most expensive operation (~50-150ms per frame on CPU)
2. **Preprocessing**: CLAHE takes ~5-10ms, denoising takes ~50-100ms
3. **Frame Processing**: Resizing, overlay, and video writing are fast (~1-5ms)

### Recommended Settings by Use Case

| Use Case | `--skip-frames` | `--detection-size` | `--no-preprocess` | Expected FPS (CPU) |
|----------|----------------|-------------------|-------------------|-------------------|
| **Maximum Speed** | 10 | 224 | Yes | 20-30+ |
| **Balanced** | 5 | 320 | Yes | 15-25 |
| **Better Accuracy** | 2 | 640 | No | 5-10 |
| **Low-Light Quality** | 3 | 416 | No | 8-15 |

### Performance Tips

- **Use GPU if available**: `--device cuda` can give 5-10x speedup
- **Lower resolution**: Smaller `--width` and `--height` = faster processing
- **Increase frame skip**: Higher `--skip-frames` = smoother playback but less accurate
- **Skip preprocessing**: Use `--no-preprocess` unless you need it for low-light
- **Lower detection size**: 224×224 is fastest, 640×640 is most accurate

---

## 🎯 Object Recognition

### Complete COCO Class List (80 Classes)

The system recognizes all 80 object classes from the COCO dataset:

**People**
- person

**Vehicles**
- bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Animals**
- bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Electronics**
- tv (monitor), laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator

**Furniture**
- chair, couch, bed, dining table, toilet

**Food & Drinks**
- bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Sports Equipment**
- sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Personal Items**
- backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard

**Other Objects**
- traffic light, fire hydrant, stop sign, parking meter, bench, potted plant, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### Verify Recognition Capabilities

Run the verification script to see all supported classes:
```bash
python -m src.verify_classes
```

This displays all 80 classes with their IDs and confirms COCO dataset coverage.

---

## 📝 Example Captions

The system generates captions based on detected interactions:

- **Single interaction**: "A person holding a bottle"
- **Using device**: "A person using a mobile phone"
- **Multiple interactions**: "A person using a laptop and a person holding a cup"
- **No interaction**: "A scene with no clear human–object interaction"
- **Complex scene**: "A person holding a bottle, a person using a keyboard, and a person using a mouse"

Captions update in real-time as interactions change in the video.

---

## 🔧 Troubleshooting

### Low FPS (< 5 FPS)
- Increase `--skip-frames` to 8-10
- Use `--no-preprocess`
- Lower `--detection-size` to 224
- Reduce `--width` and `--height`
- Check if GPU is available: `--device cuda`

### Model Not Found Error
- The model auto-downloads on first run
- If it fails, manually download: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`
- Place downloaded `yolov8n.pt` in `models/` directory

### Webcam Not Working
- Check webcam index: try `--source 1` or `--source 2`
- Verify webcam is not used by another application
- On Linux, may need camera permissions

### CUDA Out of Memory
- Use smaller `--detection-size` (224 or 320)
- Reduce frame resolution
- Use `--device cpu` if GPU memory is limited

### Poor Detection Quality
- Increase `--conf-threshold` to 0.4-0.5
- Enable preprocessing: `--preprocess`
- Use larger `--detection-size` (640)
- Reduce `--skip-frames` for more frequent detection

### Video Writer Errors
- Ensure output directory exists (auto-created)
- Check disk space for output video
- Try different output path if permissions issue

---

## 🌟 Applications

### Use Cases

1. **Assistive Technology**
   - Real-time scene descriptions for visually impaired users
   - Audio narration based on video input

2. **Surveillance & Monitoring**
   - Activity summarization for security systems
   - Automated scene understanding for operators

3. **Research & Development**
   - Human–computer interaction prototyping
   - Computer vision research platform
   - HOI (Human–Object Interaction) dataset generation

4. **Content Creation**
   - Automated video captioning
   - Accessibility features for videos
   - Dataset annotation tools

5. **Education**
   - Computer vision and deep learning demonstrations
   - Real-time object detection tutorials

---

## 🔮 Future Improvements

Potential enhancements for future versions:

- **Advanced HOI Models**: Replace heuristics with trained interaction models
- **Temporal Reasoning**: Incorporate motion tracking and temporal context
- **Action Recognition**: Detect actions (walking, sitting, eating) not just interactions
- **LLM Integration**: Use language models for more natural, diverse captions
- **Multi-Person Tracking**: Handle complex scenes with multiple people
- **Multi-Camera Support**: Process multiple video streams simultaneously
- **Uncertainty Quantification**: Show confidence scores in captions
- **Custom Class Training**: Support for fine-tuning on custom object classes
- **Web Interface**: Browser-based UI for easier interaction
- **API Server**: REST API for integration with other applications

---

## 📚 Technical Details

### Interaction Detection Algorithm

1. Detect all objects in frame using YOLOv8
2. Separate detections into:
   - **Persons**: Objects labeled "person"
   - **Objects**: All other detections
3. For each person-object pair:
   - Calculate IoU (Intersection-over-Union) of bounding boxes
   - If IoU > threshold (default: 0.1), mark as interaction
   - Classify interaction type based on object label:
     - **"using"**: phone, laptop, keyboard, mouse, remote, etc.
     - **"holding"**: bottle, cup, book, etc.
4. Generate caption from interaction list

### Performance Characteristics

- **CPU (Intel i7)**: 15-25 FPS with default settings
- **GPU (NVIDIA RTX 3060)**: 50-80 FPS with CUDA
- **Memory Usage**: ~500 MB (model) + ~200 MB (processing)
- **Model Size**: ~6 MB (yolov8n.pt)

---

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- Performance optimizations
- Additional interaction types
- Better caption generation
- Documentation improvements
- Bug fixes
- Feature requests

---

## 📄 License

This project is intended for research and educational use. Please ensure compliance with:
- **Ultralytics YOLOv8 License**: [MIT License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **OpenCV License**: [Apache 2.0](https://opencv.org/license/)

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 model and excellent documentation
- **OpenCV** for comprehensive computer vision tools
- **COCO Dataset** contributors for training data

---

## 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/nayan2723/object_detector/issues)
- **Repository**: [https://github.com/nayan2723/object_detector](https://github.com/nayan2723/object_detector)

---

## 🚦 Quick Start Summary

```bash
# 1. Clone and setup
git clone https://github.com/nayan2723/object_detector.git
cd object_detector
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Run (model auto-downloads)
python -m src.main

# 3. Verify classes
python -m src.verify_classes
```

**Enjoy real-time scene description! 🎉**

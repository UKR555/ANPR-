# ANPR (Automatic Number Plate Recognition) System

This repository contains a complete pipeline for training and using an Automatic Number Plate Recognition system based on YOLOv8.

## Features

- License plate detection in real-time video feeds
- License plate detection in video files
- License plate text extraction using OCR
- High-accuracy deep learning model based on YOLOv8
- Easy to train and use

## Requirements

- Python 3.8 or later
- CUDA-capable GPU (recommended for faster inference)
- Webcam (for real-time detection)
- Tesseract OCR (for license plate text recognition)

## Installation

1. Clone this repository
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - **Windows**: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt install tesseract-ocr`
   - **macOS**: `brew install tesseract`

4. Configure Tesseract path in `anpr_ocr.py` if needed.

## Usage

### 1. Training the Model

To train the ANPR model on your dataset:

```
python anpr_train.py
```

This will:
- Use the dataset in the Dataset folder
- Train a YOLOv8 model for license plate detection
- Save the trained model to `runs/detect/anpr_model/weights/`

### 2. Real-time License Plate Detection

To run the ANPR system with your webcam:

```
python anpr_detect_realtime.py
```

Press 'q' to quit the detection.

### 3. Processing Video Files

To process a video file:

```
python anpr_detect_video.py --video path/to/your/video.mp4
```

Additional options:
- `--model path/to/model.pt`: Use a specific model file
- `--output path/to/output.mp4`: Specify an output file path
- `--no-display`: Run without displaying the video
- `--no-save`: Don't save the output video

### 4. Detecting License Plates in Images

To detect license plates in an image:

```
python anpr_image_detection.py --image path/to/your/image.jpg
```

Additional options:
- `--model path/to/model.pt`: Use a specific model file
- `--output-dir directory`: Directory to save results
- `--conf 0.25`: Set the confidence threshold
- `--no-display`: Don't display the result
- `--no-save`: Don't save the result

### 5. OCR - Reading License Plate Text

To extract and read license plate text from an image:

```
python anpr_ocr.py --image path/to/your/image.jpg
```

Additional options:
- `--model path/to/model.pt`: Use a specific model file
- `--conf 0.4`: Set the confidence threshold
- `--output-dir ocr_results`: Directory to save results
- `--no-display`: Don't display the result

## Model Details

The ANPR model is trained to detect two classes:
- `licence`: The license plate itself
- `plate`: Full plate area

## Dataset

The dataset used for training is organized as follows:
- `Dataset/train/`: Training data
- `Dataset/valid/`: Validation data
- `Dataset/test/`: Test data

Each folder contains:
- `images/`: Image files
- `labels/`: YOLO format annotation files

## Performance

The model achieves fast inference times suitable for real-time applications:
- On GPU: 20-40 FPS depending on the hardware
- On CPU: 1-5 FPS depending on the hardware

## License

This project uses the data under CC BY 4.0 license. 
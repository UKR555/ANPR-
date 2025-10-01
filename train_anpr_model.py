from ultralytics import YOLO
import os
import yaml
import shutil
from pathlib import Path
import torch

def train_anpr_model():
    print("=== ANPR Model Training with YOLOv8 ===")

    dataset_path = Path("Dataset")
    if not dataset_path.exists():
        print("Error: Dataset directory not found.")
        return

    required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels"]
    for dir_path in required_dirs:
        if not (dataset_path / dir_path).exists():
            print(f"Error: Required directory '{dir_path}' not found.")
            return

    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        print("Error: data.yaml file not found in Dataset directory.")
        return

    print(f"Using dataset config: {data_yaml_path}")

    # Output directory
    output_dir = Path("model/anpr_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-select GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    # Load YOLOv8 base model (change to yolov8s.pt for more accuracy if you have GPU)
    model = YOLO("yolov8s.pt")

    print("Starting model training...")

    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=100,               # Train longer for better accuracy
            imgsz=640,
            batch=16,
            patience=20,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            device=device,
            project="model",
            name="anpr_model",
            exist_ok=True,
            pretrained=True,
            verbose=True,
            close_mosaic=10,         # Better for small object detection like license plates
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mixup=0.0
        )

        print("✅ Training completed successfully!")
        best_model_path = output_dir / "weights/best.pt"

        if best_model_path.exists():
            print(f"Best model saved at: {best_model_path}")
        else:
            print("⚠️ Warning: Best model not found at expected location.")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_anpr_model() 
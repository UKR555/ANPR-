from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch

# Path to the trained model
MODEL_PATH = "model/anpr_model/weights/best.pt"

def test_model_on_image(image_path):
    """Test the trained model on a single image and visualize results"""
    print(f"Testing model on image: {image_path}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    # Load the model
    try:
        # Auto-select device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        model = YOLO(MODEL_PATH)
        model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Get image dimensions
    height, width = img.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Run prediction
    try:
        # Set confidence threshold
        conf_threshold = 0.25
        print(f"Running prediction with confidence threshold: {conf_threshold}")
        
        results = model(img, conf=conf_threshold)
        
        # Process results
        result = results[0]  # Get the first result
        boxes = result.boxes
        
        print(f"Found {len(boxes)} potential license plates")
        
        # Create a copy for visualization
        vis_img = img.copy()
        
        # Process each detection
        for i, box in enumerate(boxes):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Get confidence
            conf = float(box.conf[0].item())
            
            # Get class
            cls = int(box.cls[0].item())
            
            print(f"\nDetection {i+1}:")
            print(f"  Coordinates: ({x1}, {y1}), ({x2}, {y2})")
            print(f"  Confidence: {conf:.4f}")
            print(f"  Class: {cls}")
            
            # Draw rectangle on the image
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Plate: {conf:.2f}"
            cv2.putText(vis_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Extract the plate region
            plate_img = img[y1:y2, x1:x2]
            
            # Save the plate image
            if not os.path.exists("test_results"):
                os.makedirs("test_results")
            plate_path = f"test_results/plate_{i+1}.jpg"
            cv2.imwrite(plate_path, plate_img)
            print(f"  Saved plate image to: {plate_path}")
        
        # Save the visualization
        result_path = "test_results/detection_result.jpg"
        cv2.imwrite(result_path, vis_img)
        print(f"\nSaved visualization to: {result_path}")
        
        # Display the image
        cv2.imshow("Detection Result", vis_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

# Find a test image
def find_test_image():
    """Find a suitable test image from various locations"""
    # First try to find an image in the detected_plates folder
    if os.path.exists("detected_plates"):
        files = os.listdir("detected_plates")
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join("detected_plates", file)
    
    # If no image found, try to find one in the dataset
    if os.path.exists("Dataset/test/images"):
        files = os.listdir("Dataset/test/images")
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join("Dataset/test/images", file)
    
    # If still no image found, try train images
    if os.path.exists("Dataset/train/images"):
        files = os.listdir("Dataset/train/images")
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join("Dataset/train/images", file)
    
    return None

if __name__ == "__main__":
    print("=== YOLOv8 Model Test ===")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please wait for training to complete or check the model path.")
        exit(1)
    
    # Find a test image
    test_image = find_test_image()
    if test_image is None:
        print("Error: No test image found")
        exit(1)
    
    # Test the model
    test_model_on_image(test_image) 
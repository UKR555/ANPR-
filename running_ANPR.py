import cv2
import numpy as np
import time
import os
import utils
import datetime
import torch
from ultralytics import YOLO

# Configuration
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SAVE_PATH = "detected_plates"
MODEL_PATH = "model/anpr_model/weights/best.pt"  # Path to your custom trained model

# Initialize YOLOv8 model
print("Loading custom YOLOv8 model...")
try:
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}. Please train the model first.")
        exit(1)
        
    model = YOLO(MODEL_PATH)
    print("YOLOv8 model loaded successfully")
    
    # Auto-select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load prebooked plates
with open("prebooked_plates.txt", "r") as f:
    prebooked = set(line.strip() for line in f if line.strip())
print(f"Loaded {len(prebooked)} prebooked plates")

# Create directory for saving plates
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"Created directory: {SAVE_PATH}")

def detect_and_recognize_plates():
    """Detect and recognize license plates using custom YOLOv8 model"""
    # Initialize webcam
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully. Press 'q' to quit")
    
    frame_count = 0
    last_detection_time = time.time()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Process every 5 frames to reduce CPU usage
            if frame_count % 5 == 0:
                try:
                    # Make a copy of the frame for processing
                    process_frame = frame.copy()
                    
                    # Run inference on the frame
                    results = model(process_frame, conf=0.25)
                    
                    # Process detections
                    if len(results) > 0:
                        boxes = results[0].boxes
                        
                        if len(boxes) > 0:
                            for box in boxes:
                                # Get coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                
                                # Draw rectangle
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Check if enough time has passed since last detection
                                if current_time - last_detection_time > 2:
                                    # Extract the potential plate region
                                    plate_img = process_frame[y1:y2, x1:x2]
                                    
                                    if plate_img.size > 0:
                                        # Get the predicted class and confidence
                                        plate_text = results[0].names[int(box.cls[0].item())]
                                        
                                        if plate_text:
                                            # Save the plate image
                                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                            plate_file = f"{SAVE_PATH}/plate_{timestamp}_{plate_text}.jpg"
                                            cv2.imwrite(plate_file, plate_img)
                                            
                                            # Display information
                                            print(f"\nDetected plate: {plate_text}")
                                            
                                            # Check if plate is prebooked
                                            if plate_text in prebooked:
                                                status = "✅ PREBOOKED"
                                                color = (0, 255, 0)  # Green
                                                print("✅ Prebooked. OPEN GATE.")
                                            else:
                                                status = "❌ NOT PREBOOKED"
                                                color = (0, 0, 255)  # Red
                                                print("❌ Not prebooked. DON'T OPEN GATE.")
                                            
                                            # Draw text on the frame
                                            cv2.putText(frame, plate_text, (x1, y1 - 10), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                            cv2.putText(frame, status, (x1, y2 + 20),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                            
                                            last_detection_time = current_time
                                            break  # Stop after finding one plate
                except Exception as e:
                    print(f"Error in detection: {e}")
                
                # Add FPS information
                fps = 1.0 / (time.time() - current_time + 0.001)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("ANPR System", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    print("\n=== License Plate Recognition System ===\n")
    detect_and_recognize_plates()
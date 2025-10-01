import cv2
import easyocr
import numpy as np
import time
import os
import utils
import datetime
import re

# Configuration
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SAVE_PATH = "detected_plates"

# Initialize EasyOCR
print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR initialized")

# Load prebooked plates
with open("prebooked_plates.txt", "r") as f:
    prebooked = set(line.strip() for line in f if line.strip())
print(f"Loaded {len(prebooked)} prebooked plates")

# Create directory for saving plates
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"Created directory: {SAVE_PATH}")

def is_valid_plate_format(text):
    """Check if the text matches common Indian license plate formats"""
    # Remove any spaces and convert to uppercase
    text = text.replace(" ", "").upper()
    
    # Common Indian license plate patterns:
    # 1. KA19EQ0001 (2 letters, 2 numbers, 2 letters, 4 numbers)
    # 2. KA01AB1234 (2 letters, 2 numbers, 2 letters, 4 numbers)
    # 3. KA01A1234 (2 letters, 2 numbers, 1 letter, 4 numbers)
    # 4. KA01AB123 (2 letters, 2 numbers, 2 letters, 3 numbers)
    # 5. DL7CQ1939 (2 letters, 1 number, 2 letters, 4 numbers)
    patterns = [
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # KA19EQ0001
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{3}$',  # KA01AB123
        r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',     # KA01A1234
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{2}$',  # KA01AB12
        r'^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$'   # DL7CQ1939
    ]
    
    return any(bool(re.match(pattern, text)) for pattern in patterns)

def detect_and_recognize_plates():
    """Detect and recognize license plates using contour-based detection"""
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
            
            # Process every 3 frames to improve FPS
            if frame_count % 3 == 0:
                try:
                    # Make a copy of the frame for processing
                    process_frame = frame.copy()
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Apply bilateral filter to remove noise while keeping edges
                    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
                    
                    # Find edges
                    edged = cv2.Canny(filtered, 30, 200)
                    
                    # Find contours
                    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Sort contours by area, largest first
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Reduced from 10 to 5
                    
                    # Find potential license plate contours
                    for contour in contours:
                        # Approximate the contour
                        peri = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                        
                        # License plates tend to be rectangles with 4 points
                        # Check if the contour has 4 points and is large enough
                        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                            # Get bounding box
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Check aspect ratio (license plates are usually wider than tall)
                            aspect_ratio = w / float(h)
                            if 2.0 < aspect_ratio < 6.0:
                                # This might be a license plate
                                # Draw rectangle
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                
                                # Check if enough time has passed since last detection
                                if current_time - last_detection_time > 1.5:  # Reduced from 2 to 1.5 seconds
                                    # Extract the potential plate region
                                    plate_img = process_frame[y:y+h, x:x+w]
                                    
                                    if plate_img.size > 0:
                                        # Run OCR on the potential plate
                                        ocr_results = reader.readtext(plate_img, detail=0, paragraph=False)
                                        
                                        for text in ocr_results:
                                            # Clean and normalize text
                                            plate_text = utils.clean_text(text)
                                            
                                            # Only process if the plate format is valid
                                            if plate_text and is_valid_plate_format(plate_text):
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
                                                cv2.putText(frame, plate_text, (x, y - 10), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                                cv2.putText(frame, status, (x, y + h + 20),
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
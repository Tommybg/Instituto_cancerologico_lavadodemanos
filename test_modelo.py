import torch
import cv2
import numpy as np
from pathlib import Path
from time import time
import torchvision.transforms as transforms

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define class names for your hand washing steps
CLASS_NAMES = [
    "Wet hands", "Apply soap", "Rub palms", "Rub backs", "Rub fingers", "Rinse hands"
]  # Update with your actual class names

# Load the model
def load_model(weights_path):
    try:
        # First try loading as a complete model
        model = torch.load(weights_path, map_location=device)
        print("Loaded complete model")
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading as complete model: {e}")
        print("Attempting to load with ultralytics...")
        
        # Try loading with ultralytics (for YOLO models)
        try:
            from ultralytics import YOLO
            model = YOLO(weights_path)
            print("Loaded model with ultralytics YOLO")
            return model
        except Exception as e:
            print(f"Error loading with ultralytics: {e}")
            
            # Try loading with torch hub
            try:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
                model.to(device)
                print("Loaded model with YOLOv5")
                return model
            except Exception as e:
                print(f"Error loading with torch hub: {e}")
                raise Exception("Failed to load model. Please check the model format.")

# Preprocess image for the model
def preprocess_image(img):
    # Resize to the input size expected by your model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    img_tensor = transform(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor

# Process predictions and draw on image
def process_predictions(img, predictions, model_type="standard"):
    height, width = img.shape[:2]
    annotated_img = img.copy()
    
    if model_type == "yolo":
        # For YOLO models loaded with ultralytics
        try:
            # For ultralytics YOLO models
            annotated_img = predictions[0].plot()
            return annotated_img
        except:
            # For YOLOv5 models
            try:
                return predictions.render()[0]
            except:
                print("Error rendering YOLO predictions, falling back to standard processing")
    
    # Standard processing for PyTorch models
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    
    # Try to determine the prediction format and process accordingly
    try:
        # Common YOLO format: [batch_id, class_id, confidence, x1, y1, x2, y2]
        for pred in predictions[0]:  # First batch
            if len(pred) >= 7:  # Make sure we have enough elements
                class_id = int(pred[1])
                confidence = pred[2]
                
                if confidence > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = (pred[3:7] * np.array([width, height, width, height])).astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_img, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error processing predictions: {e}")
        print("Prediction format may be different than expected")
    
    return annotated_img

# Path to your model weights
weights_path = 'weights.pt'

# Load the model
model = load_model(weights_path)

# Determine model type
model_type = "standard"
if hasattr(model, 'predict') or str(type(model)).find('YOLO') != -1:
    model_type = "yolo"
print(f"Detected model type: {model_type}")

# Function to process video
def process_video(video_path=0, output_path=None):  # 0 for webcam
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        start_time = time()
        
        if model_type == "yolo":
            # For YOLO models
            try:
                # Resize frame to 640x640 for better performance
                input_frame = cv2.resize(frame, (640, 640))
                
                # Run inference
                if hasattr(model, 'predict'):
                    predictions = model.predict(input_frame)
                else:
                    predictions = model(input_frame)
                
                # Process predictions
                annotated_frame = process_predictions(frame, predictions, model_type)
                
                # Resize back to original size if needed
                if annotated_frame.shape[:2] != (height, width):
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
            except Exception as e:
                print(f"Error during YOLO inference: {e}")
                annotated_frame = frame
        else:
            # For standard PyTorch models
            try:
                # Preprocess the frame
                img_tensor = preprocess_image(frame)
                
                # Run inference
                with torch.no_grad():
                    predictions = model(img_tensor)
                
                # Process predictions and draw on frame
                annotated_frame = process_predictions(frame, predictions, model_type)
            except Exception as e:
                print(f"Error during standard inference: {e}")
                annotated_frame = frame
        
        inference_time = time() - start_time
        
        # Add FPS info
        fps_text = f"Inference time: {inference_time:.2f}s ({1/inference_time:.2f} FPS)"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Hand Washing Detection', annotated_frame)
        
        # Write frame if writer is initialized
        if writer:
            writer.write(annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

# Run webcam detection
if __name__ == "__main__":
    # For webcam:
    process_video(0)  # 0 for webcam, output file is optional

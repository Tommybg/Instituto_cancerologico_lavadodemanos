import torch
import cv2
import numpy as np
from pathlib import Path
from time import time
import torchvision.transforms as transforms
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define class names for your hand washing steps
CLASS_NAMES = [
    "Step_1", "Step_2", "Step_3", "Step_4", "Step_5", "Step_6"
]  # Update with your actual class names

# Load the model
def load_model(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
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
                raise Exception(f"Failed to load model from {weights_path}. Please check the model format.")

# Preprocess image for the model
def preprocess_image(img, input_size=(640, 640)):
    # Resize to the input size expected by your model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
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
        except Exception as e:
            print(f"Error plotting ultralytics predictions: {e}")
            # For YOLOv5 models
            try:
                return predictions.render()[0]
            except Exception as e:
                print(f"Error rendering YOLOv5 predictions: {e}")
                print("Falling back to standard processing")
    
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
        
        # Try classification model format
        try:
            if isinstance(predictions, torch.Tensor) and predictions.ndim == 2:
                # Get the predicted class
                class_id = torch.argmax(predictions[0]).item()
                confidence = torch.softmax(predictions[0], dim=0)[class_id].item()
                
                # Add label at the top of the image
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_img, label, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing as classification model: {e}")
            print("Could not determine prediction format")
    
    return annotated_img

def main():
    # Path to your model weights
    weights_path = 'weights.pt'
    
    try:
        # Load the model
        model = load_model(weights_path)
        
        # Determine model type
        model_type = "standard"
        if hasattr(model, 'predict') or str(type(model)).find('YOLO') != -1:
            model_type = "yolo"
        print(f"Detected model type: {model_type}")
        
        # Process video from webcam
        process_video(model, model_type)
    except Exception as e:
        print(f"Error in main execution: {e}")

# Function to process video
def process_video(model, model_type, video_path=0, output_path=None):  # 0 for webcam
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video source {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Writing output to {output_path}")
    
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1
            
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
                print("User requested exit")
                break
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
    
    except Exception as e:
        print(f"Error during video processing: {e}")
    
    finally:
        # Release resources
        print("Releasing resources")
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

# Run webcam detection
if __name__ == "__main__":
    main()

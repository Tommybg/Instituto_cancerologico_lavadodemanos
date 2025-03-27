from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import torch
import torchvision.transforms as transforms
from pathlib import Path
import json
import time
 
app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")
 
# Model and device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
# Define class names for hand washing steps
CLASS_NAMES = [
    "Paso 1", "Paso 2", "Paso 3", "Paso 4", "Paso 5", "Paso 6"
]
 
# Load model (reuse code from test_modelo.py)
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
 
# Process predictions
def process_predictions(img, predictions, model_type="standard"):
    height, width = img.shape[:2]
   
    if model_type == "yolo":
        # For YOLO models loaded with ultralytics
        try:
            # Get results as dictionaries
            result = predictions[0].cpu().boxes.data.numpy()
            if len(result) > 0:
                # Just return the class with highest confidence
                class_indices = result[:, 1].astype(int)
                confidences = result[:, 4]
                max_conf_idx = np.argmax(confidences)
                class_id = int(class_indices[max_conf_idx])  # Convert int64 to Python int
                confidence = float(confidences[max_conf_idx])  # Convert to float
                return class_id, confidence
            return None, 0
        except Exception as e:
            print(f"Error processing ultralytics YOLO prediction: {e}")
            try:
                # For YOLOv5 models
                result = predictions.xyxy[0].cpu().numpy()
                if len(result) > 0:
                    # Return the class with highest confidence
                    class_indices = result[:, 5].astype(int)
                    confidences = result[:, 4]
                    max_conf_idx = np.argmax(confidences)
                    class_id = int(class_indices[max_conf_idx])  # Convert int64 to Python int
                    confidence = float(confidences[max_conf_idx])  # Convert to float
                    return class_id, confidence
                return None, 0
            except Exception as e:
                print(f"Error processing YOLOv5 prediction: {e}")
                return None, 0
   
    # Standard processing for PyTorch models
    if isinstance(predictions, torch.Tensor):
        try:
            preds = predictions.cpu().detach().numpy()
            # Assuming predictions are class probabilities
            if len(preds.shape) > 1 and preds.shape[1] >= len(CLASS_NAMES):
                class_id = int(np.argmax(preds[0]))  # Convert int64 to Python int
                confidence = float(preds[0][class_id])  # Convert to float
                return class_id, confidence
        except Exception as e:
            print(f"Error processing standard prediction: {e}")
   
    return None, 0
 
# Load the model
weights_path = 'weights.pt'
model = load_model(weights_path)
 
# Determine model type
model_type = "standard"
if hasattr(model, 'predict') or str(type(model)).find('YOLO') != -1:
    model_type = "yolo"
print(f"Detected model type: {model_type}")
 
@app.route('/')
def index():
    return render_template('index.html')
 
@socketio.on('connect')
def handle_connect():
    print('Client connected')
 
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
 
# Store last few detections to smooth out results
last_detections = []
DETECTION_HISTORY_SIZE = 5
CONFIDENCE_THRESHOLD = 0.6  # Lower threshold to catch more detections
 
@socketio.on('video_frame')
def handle_video_frame(data):
    global last_detections
   
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        bounding_boxes = []
       
        # Run inference
        if model_type == "yolo":
            # Resize frame for better performance
            input_frame = cv2.resize(frame, (640, 640))
           
            # Run inference
            if hasattr(model, 'predict'):
                predictions = model.predict(input_frame)
            else:
                predictions = model(input_frame)
           
            # Process predictions
            class_id, confidence = process_predictions(frame, predictions, model_type)
           
            # Extract and draw bounding boxes on the frame
            if hasattr(predictions[0], 'boxes') and hasattr(predictions[0].boxes, 'data'):
                for box in predictions[0].cpu().boxes.data.numpy():
                    x1, y1, x2, y2, conf, cls = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_id = int(cls)
                   
                    # Store bounding box info
                    bounding_boxes.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'class': class_id,
                        'label': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}",
                        'confidence': float(conf)
                    })
                   
                    # Draw on the display frame
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f'{CLASS_NAMES[class_id]}: {conf:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Preprocess the frame
            img_tensor = preprocess_image(frame)
           
            # Run inference
            with torch.no_grad():
                predictions = model(img_tensor)
           
            # Process predictions
            class_id, confidence = process_predictions(frame, predictions, model_type)
       
        # Update detection history
        if class_id is not None and confidence > CONFIDENCE_THRESHOLD:
            last_detections.append((class_id, confidence))
        else:
            # If no detection, add a None to history
            last_detections.append((None, 0))
           
        # Keep history at fixed size
        if len(last_detections) > DETECTION_HISTORY_SIZE:
            last_detections = last_detections[-DETECTION_HISTORY_SIZE:]
           
        # Get most common detection in recent history
        valid_detections = [(c, conf) for c, conf in last_detections if c is not None]
       
        if valid_detections:
            # Count occurrences of each class
            class_counts = {}
            for c, conf in valid_detections:
                class_counts[c] = class_counts.get(c, 0) + 1
               
            # Get the most common class
            most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
           
            # Get the highest confidence for that class
            max_confidence = max([conf for c, conf in valid_detections if c == most_common_class])
           
            # Use the most common class from recent history
            class_id = most_common_class
            confidence = max_confidence
        else:
            class_id = None
            confidence = 0
       
        # Convert the frame with bounding boxes back to base64
        _, buffer = cv2.imencode('.jpg', display_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
       
        # Send result back to client
        if class_id is not None:
            step_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
            print(f"Detected: {step_name} with confidence {confidence:.2f}")
            emit('prediction_result', {
                'step': int(class_id),
                'step_name': step_name,
                'confidence': float(confidence),
                'image': f'data:image/jpeg;base64,{img_base64}',
                'bounding_boxes': bounding_boxes
            })
        else:
            emit('prediction_result', {
                'step': -1,
                'step_name': 'No step detected',
                'confidence': 0,
                'image': f'data:image/jpeg;base64,{img_base64}',
                'bounding_boxes': []
            })
   
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        emit('prediction_result', {'step': -1, 'step_name': 'Error', 'confidence': 0})
 
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
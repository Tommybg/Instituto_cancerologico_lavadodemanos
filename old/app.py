from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from test_modelo import load_model, process_predictions

app = Flask(__name__)
CORS(app)

# Load model once when starting server
model = load_model('weights.pt')
REQUIRED_TIME_PER_STEP = 7  # seconds

@app.route('/detect-step', methods=['POST'])
def detect_step():
    try:
        # Get image data from request
        image_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        
        # Convert to opencv format
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        predictions = model.predict(frame)
        
        # Get the detected step (you'll need to modify this based on your model's output format)
        detected_step = process_predictions(frame, predictions, return_class=True)
        
        return jsonify({
            'detected_step': detected_step,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
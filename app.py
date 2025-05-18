from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os
import cv2
import tempfile
from tensorflow.keras.models import load_model
from flask_cors import CORS
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='frontend')
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["POST", "OPTIONS", "GET"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Load the model
model = None
MODEL_PATH = 'glasses_model.h5'

def load_ml_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at path: {MODEL_PATH}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Directory contents: {os.listdir('.')}")
            return False
            
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Directory contents: {os.listdir('.')}")
        logger.error(traceback.format_exc())
        return False

# Serve frontend static files
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

# API endpoint for glasses detection
@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS':
        return '', 204

    logger.info("Received detection request")
    
    if model is None:
        logger.info("Loading model for the first time")
        if not load_ml_model():
            return jsonify({
                'error': 'Failed to load ML model. Please try again later.',
                'details': 'Model initialization error'
            }), 500
        
    if 'image' not in request.files:
        logger.warning("No image file in request")
        return jsonify({'prediction': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({'prediction': 'No image selected'}), 400

    try:
        logger.info(f"Processing image: {file.filename}")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            file.save(temp.name)
            img_path = temp.name
            logger.debug(f"Saved temporary file at: {img_path}")

        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            logger.error("Failed to read image with OpenCV")
            return jsonify({'prediction': 'Invalid image format'}), 400
            
        logger.debug(f"Original image shape: {img.shape}")
        img = cv2.resize(img, (64, 64))
        logger.debug(f"Resized image shape: {img.shape}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        logger.info("Making prediction")
        prediction = model.predict(img)
        logger.debug(f"Raw prediction value: {prediction}")

        os.remove(img_path)
        logger.debug("Temporary file removed")

        prob = prediction[0][0]
        result = "Wearing Glasses" if prob > 0.5 else "Not Wearing Glasses"
        confidence = f"{prob * 100:.2f}% confidence"

        response_data = {
            'prediction': f"{result} ({confidence})",
            'probability': float(prob),
            'wearing_glasses': bool(prob > 0.5)
        }
        logger.info(f"Sending response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'prediction': f"Error: {str(e)}",
            'error_details': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'status': 'healthy',
        'model_loaded': model is not None
    }
    return jsonify(status), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    load_ml_model()
    app.run(host='0.0.0.0', port=port)
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os
import cv2
import tempfile
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__, static_folder='frontend')
CORS(app, resources={
    r"/detect": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5000",
            "https://your-vercel-app-url.vercel.app"  # Update this with your Vercel URL
        ]
    }
})

# Load the model
model = None

def load_ml_model():
    global model
    model = load_model('glasses_model.h5')

# Serve frontend static files
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

# API endpoint for glasses detection
@app.route('/detect', methods=['POST'])
def detect():
    if model is None:
        load_ml_model()
        
    if 'image' not in request.files:
        return jsonify({'prediction': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'prediction': 'No image selected'}), 400

    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            file.save(temp.name)
            img_path = temp.name

        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            return jsonify({'prediction': 'Invalid image format'}), 400
            
        img = cv2.resize(img, (64, 64))  # match model input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        img = img / 255.0  # normalize
        img = np.expand_dims(img, axis=0)  # shape becomes (1, 64, 64, 3)

        prediction = model.predict(img)
        os.remove(img_path)

        prob = prediction[0][0]
        result = "Wearing Glasses" if prob > 0.5 else "Not Wearing Glasses"
        confidence = f"{prob * 100:.2f}% confidence"

        return jsonify({
            'prediction': f"{result} ({confidence})",
            'probability': float(prob),
            'wearing_glasses': bool(prob > 0.5)
        })

    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    load_ml_model()
    app.run(host='0.0.0.0', port=port)
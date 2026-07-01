# import necessary libraries.
import os
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask App
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/US_FetalPlane_Detection_Model.keras'
IMAGE_SIZE = (224, 224)

# Load trained Ultrasound Fetal Plane Detection CNN model
model = load_model(MODEL_PATH, compile=False)

# Define fetal plane class labels
CLASSES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix']

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img = img.convert("RGB")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            # Process image and predict
            processed_image = preprocess_image(filepath)
            predictions = model.predict(processed_image)
            
            # Get class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_class_idx]
            confidence = float(np.max(predictions[0]))

            return jsonify({
                'class': predicted_class,
                'confidence': confidence,
                'image_url': filepath
            })
        except Exception as e:
            return jsonify({'error': str(e)})
        

if __name__ == '__main__':
    app.run(debug=True)
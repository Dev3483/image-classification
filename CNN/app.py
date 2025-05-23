from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

import os




app = Flask(__name__)

model = load_model('CNN/cnn_cifar10_model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert('RGB').resize((32, 32))    
    img = np.array(img) / 255.0
    img = img.reshape(1, 32, 32, 3)
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    return jsonify({'prediction': class_names[class_id]})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var on Render, or 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)

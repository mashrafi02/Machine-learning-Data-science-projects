from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow import keras
import base64

MODEL =  keras.models.load_model('../models/model.keras')
MODEL_LEAF = keras.models.load_model('../models/model_leaf_detection.keras')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']
CLASS_LABEL = ['Not Potato leaf', 'Potato leaf']

app = Flask(__name__)

def read_file_as_image(data) -> np.ndarray:
    image_array = np.array(Image.open(BytesIO(data)))
    return image_array


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        file = request.files['file']

        image = read_file_as_image(file.read())

        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_src = f"data:image/jpeg;base64,{image_data}"
        
        image_batch = np.expand_dims(image,0)

        predictions = MODEL.predict(image_batch)
        predictions_isLeaf = MODEL_LEAF.predict(image_batch)

        class_label = CLASS_NAMES[np.argmax(predictions[0])]
        class_label_isLeaf = CLASS_LABEL[np.argmax(predictions_isLeaf[0])]

        confidence = round(100 * np.max(predictions[0]), 2)

        result = None
        if class_label_isLeaf == 'Potato leaf' and confidence > 70:
            return render_template('index.html', class_label=class_label, result=confidence, image=image_src)
        else:
            comment = 'Please enter a valid and clear Potato leaf'
            return render_template('index.html', comment = comment, image=image_src, result= result )
        
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
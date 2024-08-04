from flask import Flask, render_template, request, redirect, url_for 
import joblib
import json
import numpy as np
from necessary_functions import get_cropped_image_for_2_eyes, get_the_features
from flask_sqlalchemy import SQLAlchemy 
import base64


np.set_printoptions(suppress=True)

with open('artifacts/player_dictionary.json', 'r') as file:
    data = json.load(file)

model = joblib.load('artifacts/model.pkl')

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///image_files.db'

db = SQLAlchemy(app)
# db.init_app(app)


class Upload(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    filename = db.Column(db.String(100), nullable = False)
    data = db.Column(db.LargeBinary)
    cropped_data = db.Column(db.LargeBinary)

with app.app_context():
    db.create_all()


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == "POST":
        file = request.files['file']

        image_array = np.frombuffer(file.read(), np.uint8)
        cropped_image = get_cropped_image_for_2_eyes(image_array)

        upload = Upload(filename=file.filename, data=image_array.tobytes(), cropped_data = cropped_image)
        db.session.add(upload)
        db.session.commit()

        return redirect(url_for('get_result', name=file.filename))
    return render_template('index.html')
        

@app.route('/results/<name>', methods=['GET','POST'])
def get_result(name):
    file_data = db.session.execute(db.select(Upload).where(Upload.filename == name)).scalar()

    image_data = base64.b64encode(file_data.data).decode('utf-8')
    image_src = f"data:image/jpeg;base64,{image_data}"

    if file_data.cropped_data is not None:
        cropped_array = np.frombuffer(file_data.cropped_data, dtype=np.uint8)

        player_probability_dict = {}

        X = get_the_features(cropped_array)
        X = np.array(X).reshape(len(X), 4096).astype('float')

        prediction = model.predict(X)[0]
        result_proba = model.predict_proba(X)[0] * 100


        result = None
        if max(result_proba) > 70:
            for key, value in data.items():
                if value == prediction:
                    result = key.title()
                    break
        else:
            result = "Unidentified"

        count = 0
        for player, number in data.items():
            player_probability_dict[player] = round(result_proba[count], 4)
            count += 1

        return render_template('index.html', result=result, player_probability_dict=player_probability_dict, image = image_src)
    
    return "The image you imported the face isn't clear much. Try with another one by going back!!!"


if __name__ == "__main__":
    app.run(debug=True)

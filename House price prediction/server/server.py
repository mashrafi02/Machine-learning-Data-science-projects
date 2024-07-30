from flask import Flask, request, jsonify, render_template
import json
import pickle
import numpy as np

with open('artifacts/columns.json', 'r') as file:
    data = json.load(file)
with open('artifacts/bengaluru_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    location = data['data_columns'][3:]

    estimate_price = None
    sqft = bathrooms = bedrooms = location_get = ""
    if request.method == "POST":
        location_get = request.form["location"]
        loc_index = data['data_columns'].index(location_get)
        x = np.zeros(len(data['data_columns']))

        x[0] = request.form["sqft"]
        x[1] = request.form["bathrooms"]
        x[2] = request.form["bedrooms"]
        if loc_index > 0:
            x[loc_index] = 1
        estimate_price =  round((model.predict([x])[0])*100000, 2)

    return render_template('index.html', locations=location, price=estimate_price, sqft=sqft, bathrooms=bathrooms, bedrooms=bedrooms, location_get=location_get)

if __name__ == "__main__":
    app.run(debug=False)
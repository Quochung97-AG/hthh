#import libraries

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model_data = pickle.load(open('model.pkl', 'rb'))
loaded_model = model_data["model"]
label_encoder = model_data["label_encoder"]


#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():

    try:
        # Retrieve input values from the form
        sepal_length = request.form.get('sepal.length')
        sepal_width = request.form.get('sepal.width')
        petal_length = request.form.get('petal.length')
        petal_width = request.form.get('petal.width')

        # Check if all values are provided
        if not sepal_length or not sepal_width or not petal_length or not petal_width:
            return render_template('index.html', prediction_text="Vui lòng nhập đầy đủ thông tin.")

        # Convert values to float and validate
        try:
            sepal_length = float(sepal_length)
            sepal_width = float(sepal_width)
            petal_length = float(petal_length)
            petal_width = float(petal_width)
        except ValueError:
            return render_template('index.html', prediction_text="Xin hãy nhập đúng và hợp lệ.")

        # Prepare input data for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict using the loaded model
        predicted_variety_numeric = loaded_model.predict(input_data)
        predicted_variety = label_encoder.inverse_transform([int(round(predicted_variety_numeric[0]))])

        # Render the result
        return render_template('index.html', prediction_text=f"Dự đoán hoa iris là: {predicted_variety[0]}")

    except Exception as e:
        # Handle unexpected errors
        return render_template('index.html', prediction_text=f"Lỗi hệ thống !: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
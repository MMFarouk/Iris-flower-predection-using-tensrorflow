from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__, template_folder='templates')

def load_model():
    global model
    model = tf.saved_model.load('saved_model')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def predict():
    if 'model' not in globals():
        load_model()

    # Get input values from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Prepare the input data for prediction
    to_predict = np.array([sepal_length, sepal_width, petal_length, petal_width], dtype=np.float32)
    input_data = {'input_1': np.array([to_predict])}  # Use the correct input name from your model

    # Make predictions using the TensorFlow model
    result = model.signatures['serving_default'](**input_data)
    keys = result.keys()  # Get the keys for the model's outputs
    prediction = result[list(keys)[0]].numpy()  # Assuming the first key contains the output data

    # Convert the class to Iris species name
    species = ['Setosa', 'Versicolor', 'Virginica'][np.argmax(prediction)]

    # Place the result in the response
    return render_template('home.html', result=species)  

if __name__ == '__main__':
    app.run(debug=True)
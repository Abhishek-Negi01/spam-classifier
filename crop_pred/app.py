import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
with open(r'model\model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the feature columns that your model was trained on
model_columns = model.feature_names_in_  # The column names expected by your model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the HTML form
    crop_year = int(request.form['year'])
    area = float(request.form['Area'])
    production = float(request.form['production'])
    rainfall = float(request.form['Annual_Rainfall'])
    fertilizer = float(request.form['Fertilizer'])
    pesticide = float(request.form['Pesticide'])
    season = request.form['season']
    state = request.form['state']

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Crop_Year': [crop_year],
        'Area': [area],
        'Production': [production],
        'Annual_Rainfall': [rainfall],
        'Fertilizer': [fertilizer],
        'Pesticide': [pesticide],
        'Season': [season],
        'State': [state],
    })

    # Apply one-hot encoding to categorical columns (e.g., Season, State)
    input_data_encoded = pd.get_dummies(input_data)

    # Ensure that the input data matches the model’s expected feature columns
    input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction using the trained model
    prediction = model.predict(input_data_encoded)

    # Render the result on a new page
    return render_template('result.html', prediction=prediction[0])


@app.route('/new_prediction')
def new_prediction():
    # Render the index page for a new prediction
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

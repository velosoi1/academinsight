import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Load the CSV data into a Pandas DataFrame
data = pd.read_csv('bibertrain.csv')

# Extract features (numerical frequencies) and target for communicative purpose
X_train_purpose = data.iloc[:, 3:]  # Assuming feature columns start from the fourth column
y_train_purpose = LabelEncoder().fit_transform(data['main_full'])

# Extract features and target for discipline
X_train_discipline = data.iloc[:, 3:]  # Assuming feature columns start from the fourth column
y_train_discipline = LabelEncoder().fit_transform(data['discipline'])

# Create and train logistic regression models for communicative purpose and discipline
model_purpose = LogisticRegression()
model_purpose.fit(X_train_purpose, y_train_purpose)

model_discipline = LogisticRegression()
model_discipline.fit(X_train_discipline, y_train_discipline)

# Create and fit label encoders for inverse transformation
label_encoder_purpose = LabelEncoder()
label_encoder_purpose.fit(data['main_full'])

label_encoder_discipline = LabelEncoder()
label_encoder_discipline.fit(data['discipline'])

# Create and fit scalers for scaling input data
scaler_purpose = StandardScaler()
scaler_purpose.fit(X_train_purpose)

scaler_discipline = StandardScaler()
scaler_discipline.fit(X_train_discipline)

app = Flask(__name__, template_folder='./')

@app.route('/')
def choose_model():
    return render_template('choose_model.html')

# Route for about page
@app.route('/about_page', methods=['GET'])
def about_page():
    return render_template('about_page_biber.html')

@app.route('/predict_communicative_purpose', methods=['GET', 'POST'])
def predict_communicative_purpose():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            try:
                input_data = pd.read_csv(uploaded_file)

                predictions = []

                for index, row in input_data.iterrows():
                    # Extract the file name for the current row
                    file_name = row.iloc[0]

                    # Extract features for the current row
                    features = row.iloc[1:].values.reshape(1, -1)

                    # Scale the input features using the same scaler
                    features_scaled = scaler_purpose.transform(features)

                    # Predict the communicative purpose for the current row
                    prediction_probabilities = model_purpose.predict_proba(features_scaled)[0]

                    # Get the top communicative purpose and its probability
                    top_communicative_purpose_index = prediction_probabilities.argmax()
                    top_communicative_purpose = label_encoder_purpose.inverse_transform([top_communicative_purpose_index])[0]
                    top_probability = prediction_probabilities[top_communicative_purpose_index]

                    # Get the percentages for all communicative purposes
                    percentages_purpose = {label: probability * 100 for label, probability in zip(model_purpose.classes_, prediction_probabilities)}

                    # Extract the top 5 tags for the current row
                    coefficients = model_purpose.coef_[top_communicative_purpose_index]
                    top_tags_indices = coefficients.argsort()[-5:][::-1]  # Get the indices of top 5 tags
                    top_tags = [data.columns[i+3] for i in top_tags_indices]  # Assuming tags are in columns 4 onwards

                    # Store the prediction results for the current row
                    predictions.append({
                        'file_name': file_name,
                        'communicative_purpose': top_communicative_purpose,
                        'probability': top_probability,
                        'percentages': percentages_purpose,
                        'top_tags': top_tags
                    })

                return render_template('index_cp.html', predictions=predictions)
            except Exception as e:
                logging.error(f"Prediction Error: {str(e)}")
                return render_template('index_cp.html', error_message=str(e))
    
    return render_template('index_cp.html', predictions=None, error_message=None)

@app.route('/predict_discipline', methods=['GET', 'POST'])
def predict_discipline():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            try:
                input_data = pd.read_csv(uploaded_file)

                predictions = []

                for index, row in input_data.iterrows():
                    # Extract the file name from the first column
                    file_name = row.iloc[0]

                    # Extract features from the current row
                    input_features = row.iloc[1:]

                    # Reshape features to match the shape expected by the scaler
                    input_features = input_features.values.reshape(1, -1)

                    # Scale the input features using the same scaler
                    input_features_scaled = scaler_discipline.transform(input_features)

                    # Predict the discipline based on the uploaded features
                    prediction_probabilities = model_discipline.predict_proba(input_features_scaled)[0]

                    # Get the top discipline and its probability
                    top_discipline_index = prediction_probabilities.argmax()
                    top_discipline = label_encoder_discipline.inverse_transform([top_discipline_index])[0]
                    top_probability = prediction_probabilities[top_discipline_index]

                    # Get the percentages for all disciplines
                    percentages_discipline = {label: probability * 100 for label, probability in zip(model_discipline.classes_, prediction_probabilities)}

                    # Get the top 5 tags most indicative of the predicted discipline
                    top_tags = [tag for tag, coef in sorted(zip(X_train_discipline.columns, model_discipline.coef_[top_discipline_index]), key=lambda x: abs(x[1]), reverse=True)][:5]

                    predictions.append({
                        'file_name': file_name,
                        'discipline': top_discipline,
                        'probability': top_probability,
                        'percentages': percentages_discipline,
                        'top_tags': top_tags
                    })

                logging.debug("Input Data:")
                logging.debug(input_data)
                logging.debug("Predictions:")
                logging.debug(predictions)

                return render_template('index_discipline.html', predictions=predictions)
            except Exception as e:
                logging.error(f"Prediction Error: {str(e)}")
                return render_template('index_discipline.html', error_message=str(e))
    
    return render_template('index_discipline.html', predictions=None, error_message=None)

if __name__ == '__main__':
    app.run(host='localhost', port=5025)

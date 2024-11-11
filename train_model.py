from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

app = Flask(__name__)

# Train the model and save it
def train_model():
    # Load dataset
    data = pd.read_csv('marketing_recommendation.csv', encoding='ISO-8859-1') 

    # Encode categorical variables
    categorical_columns = ['Country
    for column in categorical_columns:
        data[column] = data[column].astype('category').cat.codes  # Encoding categorical variables

    # Prepare feature and target variables
    X = data[['UnitPrice', 'Quantity', 'Country']]  # Selected features
    y = data['Quantity']  # Target variable - customer purchase quantity

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save the model
    joblib.dump(model, 'recommendation_model.pkl')
    print("Model saved as 'recommendation_model.pkl'")
    return mse, r2

# Route to load the index page
@app.route('/')
def index():
    mse, r2 = train_model()  # Train and evaluate model
    return render_template('index.html', mse=mse, r2=r2)

# Route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Load the saved model
    model = joblib.load('recommendation_model.pkl')

    # Get user input
    unit_price = float(request.form['UnitPrice'])
    quantity = int(request.form['Quantity'])
    country = int(request.form['Country'])

    # Make prediction
    input_data = pd.DataFrame([[unit_price, quantity, country]], columns=['UnitPrice', 'Quantity', 'Country'])
    prediction = model.predict(input_data)[0]
    
    return jsonify({'predicted_quantity': prediction})

if __name__ == '__main__':
    app.run(debug=True)

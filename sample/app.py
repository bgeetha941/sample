from flask import Flask, jsonify, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# HTML templates as string inside Python
dataset_upload_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Upload</title>
</head>
<body>
    <h2>Upload Dataset</h2>
    <form action="/upload_dataset" method="POST" enctype="multipart/form-data">
        <label for="dataset">Upload CSV Dataset:</label><br>
        <input type="file" id="dataset" name="dataset" accept=".csv"><br><br>
        <input type="submit" value="Submit Dataset">
    </form>
</body>
</html>
"""

domain_selection_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Select Domain</title>
</head>
<body>
    <h2>Select Domain</h2>
    <form action="/select_domain" method="POST">
        <label for="domain">Choose a domain:</label><br>
        <select id="domain" name="domain">
            <option value="Finance">Finance</option>
            <option value="Healthcare">Healthcare</option>
            <option value="Retail">Retail</option>
            <option value="Education">Education</option>
            <option value="Manufacturing">Manufacturing</option>
            <option value="Telecommunications">Telecommunications</option>
        </select><br><br>
        <input type="submit" value="Submit Domain">
    </form>
</body>
</html>
"""

result_page_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results</title>
</head>
<body>
    <h2>Results for Domain: {{ domain }}</h2>
    <p><strong>Best Model:</strong> {{ best_model }}</p>
    <p><strong>Domain-specific Recommendation:</strong> {{ domain_recommendation }}</p>
    <h3>Model Performance</h3>
    {% for model, metrics in results.items() %}
    <h4>Model: {{ model }}</h4>
    <ul>
    {% for metric, value in metrics.items() %}
        <li>{{ metric }}: {{ value }}</li>
    {% endfor %}
    </ul>
    {% endfor %}
</body>
</html>
"""

# In-memory storage of dataset
dataset = None

# Research-based domain recommendations
def research_based_recommendations(domain):
    domain_models = {
        "Finance": "RandomForest",
        "Healthcare": "NeuralNetwork",
        "Retail": "RandomForest",
        "Education": "NeuralNetwork",
        "Manufacturing": "SVM",
        "Telecommunications": "SVM",
    }
    return domain_models.get(domain, "No specific model recommendation available for this domain.")

# Data Preprocessing
def preprocess_data(df):
    # Handle missing values by filling them with the mean (or other strategies)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Handle categorical columns using Label Encoding
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders

def load_and_preprocess_data(df):
    df, label_encoders = preprocess_data(df)
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate model
def evaluate_model(y_test, y_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    }

# Train traditional ML models
def train_traditional_models(X_train, y_train, X_test, y_test):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier()  
    }
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = evaluate_model(y_test, y_pred)
    return results

@app.route('/')
def index():
    return render_template_string(dataset_upload_page)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global dataset
    file = request.files['dataset']
    
    try:
        dataset = pd.read_csv(file)
        return render_template_string(domain_selection_page)
    except Exception as e:
        return f"Error processing the dataset: {str(e)}", 400

@app.route('/select_domain', methods=['POST'])
def select_domain():
    global dataset
    if dataset is None:
        return "No dataset uploaded.", 400

    domain = request.form.get('domain')

    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset)

        # Train models and get results
        model_results = train_traditional_models(X_train, y_train, X_test, y_test)
        best_model = max(model_results, key=lambda m: model_results[m]['Accuracy'])

        # Get domain-specific recommendation
        domain_recommendation = research_based_recommendations(domain)

        # Render results dynamically
        return render_template_string(result_page_template, 
                                      domain=domain, 
                                      best_model=best_model,
                                      domain_recommendation=domain_recommendation,
                                      results=model_results)
    except Exception as e:
        return f"Error processing the dataset or domain: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)

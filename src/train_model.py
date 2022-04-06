# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from joblib import dump
from data import process_data
from model import train_model, compute_model_metrics, inference
from evaluate_model_slices import category_slice_metrics


# Add code to load in the data.
data = pd.read_csv('data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
dump(model, 'model/rfc_model.joblib')
dump(encoder, 'model/encoder.joblib')
dump(lb, 'model/lb.joblib')

preds_test = inference(model, X_test)

# Calculate and save model metrics.
precision, recall, fbeta = compute_model_metrics(y_test, preds_test)
metrics_dict = {
    'precision': precision,
    'recall': recall,
    'fbeta': fbeta
}

with open('metrics/model_metrics.txt', 'w') as filename:
    filename.write(str(metrics_dict))

# Calculate and save model metrics for each category slice.
category_slice_metrics(test, cat_features, y_test, preds_test)


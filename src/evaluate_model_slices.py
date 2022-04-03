import numpy as np
import pandas as pd
from joblib import load
from data import process_data
from sklearn.model_selection import train_test_split
from model import compute_model_metrics, inference


rfc_model = load('../model/rfc_model.joblib')
encoder = load('../model/encoder.joblib')
lb = load('../model/lb.joblib')

data = pd.read_csv('../data/census_clean.csv')

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


X_test, y_test, encoder_test, lb_test = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

preds = inference(rfc_model, X_test)


def category_slice_metrics(df, cat_features, y, preds):
    '''
    Validates the trained machine learning model on
    categorical slices using precision, recall, and F1.
    
    Inputs
    ------
    df : pd.DataFrame
        Test dataset
    cat_features : list
        Categorical features.
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    '''
    cat_df = df[cat_features].copy()
    cat_df['y'] = y.tolist()
    cat_df['preds'] = preds.tolist()
    
    category_list = []
    category_slice_list = []
    precision_list = []
    recall_list = []
    fbeta_list = []
    num_records_list = []
    
    for cat_feature in cat_features:
        cat_slices = df[cat_feature].unique()
        for cat_slice in cat_slices:
            slice_df = cat_df[cat_df[cat_feature] == cat_slice]
            slice_y = np.array(slice_df['y'].to_list())
            slice_preds = np.array(slice_df['preds'].to_list())
            precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
            
            category_list.append(cat_feature)
            category_slice_list.append(cat_slice)
            precision_list.append(precision)
            recall_list.append(recall)
            fbeta_list.append(fbeta)
            num_records_list.append(len(slice_df))
    
    slice_metrics_dict = {
        'category': category_list,
        'category_slice': category_slice_list,
        'precision': precision_list,
        'recall': recall_list,
        'fbeta': fbeta_list,
        'num_records': num_records_list
    }
    
    slice_metrics_df = pd.DataFrame.from_dict(slice_metrics_dict)
    
    with open('../metrics/slice_output.txt', 'w') as filename:
        filename.write(slice_metrics_df.to_string(index=False))


if __name__ == "__main__":
    category_slice_metrics(test, cat_features, y_test, preds)
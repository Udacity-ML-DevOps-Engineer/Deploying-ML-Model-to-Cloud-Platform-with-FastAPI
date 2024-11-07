import pandas as pd
import pickle
import os
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slice_metrics
from sklearn.model_selection import train_test_split

def train_and_evaluate_model(data_path):
    # Create the model directory
    os.makedirs("../model", exist_ok=True)

    # Define the model path
    model_path = "../model/model.pkl"
    encoder_path = "../model/encoder.pkl"
    lb_path = "../model/lb.pkl"
    
    # Load the data from the csv file.
    data = pd.read_csv(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    # Save the model, encoder, and lb using pickle.
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    with open(lb_path, "wb") as f:
        pickle.dump(lb, f)

    # Predict on the test data.
    y_pred = inference(model, X_test)

    # Calculate the model metrics.
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    # Print the model metrics.
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-beta: {fbeta}")

    # Compute slice metrics.
    for feature in cat_features:
        metrics_df = compute_slice_metrics(
            data=test,
            feature=feature,
            model=model,
            encoder=encoder,
            lb=lb,
            cat_features=cat_features
        )
        print(f"Metrics for feature: {feature}")
        metrics_df.to_csv(f"./metrics/slice_output_{feature}.txt", index=False)


if __name__ == "__main__":
    data_path = "./data/census_cleaned.csv"
    train_and_evaluate_model(data_path)


import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import json

def preprocess_data(df):
    """
    Preprocess the dataset:
    - Convert 'references' to a numeric feature 'times_of_references'.
    - Drop unused columns.
    """
    df["times_of_references"] = df["references"].apply(len)
    return df.drop(columns=["references", "id"], errors="ignore")

def main():
    logging.getLogger().setLevel(logging.INFO)

    # Load datasets
    train_file = "train.json"
    test_file = "test.json"

    train = pd.read_json(train_file)
    test = pd.read_json(test_file)

    # Preprocess datasets
    train = preprocess_data(train)
    test = preprocess_data(test)

    # Split the data
    train, validation = train_test_split(train, test_size=1 / 3, random_state=123)

    # Feature extraction
    numeric_features = ["year", "times_of_references"]
    featurizer = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("abstract_tfidf", TfidfVectorizer(max_features=4000), "abstract"),
            ("title_tfidf", TfidfVectorizer(max_features=1000), "title"),
            ("authors_tfidf", TfidfVectorizer(max_features=1000), "authors"),
        ],
        remainder="drop"
    )
    pipeline = Pipeline([
        ("featurizer", featurizer),
        ("lgbm", LGBMRegressor(
            num_leaves=300,
            n_estimators=250,
            max_depth=50,
            learning_rate=0.1,
            random_state=123
        ))
    ])


    label = "n_citation"

    # Train the model
    pipeline.fit(train.drop(columns=[label]), np.log1p(train[label]))

    # Calculate MAE for training and validation sets
    train_mae = mean_absolute_error(
        train[label], np.expm1(pipeline.predict(train.drop(columns=[label])))
    )
    validation_mae = mean_absolute_error(
        validation[label], np.expm1(pipeline.predict(validation.drop(columns=[label])))
    )

    # Log results
    logging.info(f"LightGBM train MAE: {train_mae:.2f}")
    logging.info(f"LightGBM validation MAE: {validation_mae:.2f}")

    # Make predictions on the test set
    test_predictions = np.expm1(pipeline.predict(test))  # Reverse log-transform

    # Ensure predictions are in the same order as test.json
    predicted_list = [{"n_citation": float(pred)} for pred in test_predictions]

    # Save predictions to predicted.json
    predicted_file = "predicted.json"
    with open(predicted_file, "w") as f:
        json.dump(predicted_list, f, indent=2)

    logging.info(f"Predictions saved to {predicted_file}")

if __name__ == "__main__":
    main()


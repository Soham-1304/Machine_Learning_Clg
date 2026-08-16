import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ml_utils  # our own module


def main():
    # Load a sample dataset (Iris — classic classification dataset)
    iris = load_iris(as_frame=True)
    df = iris.frame  # includes features + a 'target' column

    print("=== Step 1: Quick Summary ===")
    ml_utils.quick_summary(df)

    print("\n=== Step 2: Train/Test Split ===")
    X_train, X_test, y_train, y_test = ml_utils.split_data(df, target_column="target")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("\n=== Step 3: Scale Features ===")
    X_train_scaled, X_test_scaled, scaler = ml_utils.scale_features(X_train, X_test)
    print("Scaling done. First scaled training row:", X_train_scaled[0])

    print("\n=== Step 4: Train a model + Evaluate ===")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = ml_utils.evaluate_model(y_test, y_pred, task="classification")
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
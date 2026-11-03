"""Train a small classifier and log it with Triton serving files.

Install: pip install mlflow-triton scikit-learn
Run from the repository root: python examples/quickstart.py

Uses scikit-learn's bundled Iris dataset. MLflow uses your configured tracking
destination, or its local default. No Triton server is required for logging.
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlflow_triton import sklearn


def main():
    inputs, targets = load_iris(return_X_y=True, as_frame=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    model.fit(inputs, targets)

    info = sklearn.log_model(
        model,
        name="iris_classifier",
        input_example=inputs.iloc[:2],
    )
    print(f"Model and Triton serving files saved to: {info.model_uri}")


if __name__ == "__main__":
    main()

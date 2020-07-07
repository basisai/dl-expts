"""
Script to train model.
"""
import logging
import pickle

from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.analyzer import ModelTypes
from bedrock_client.bedrock.api import BedrockApi
import lightgbm as lgb
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from constants import FEATURES, TARGET, CONFIG_FAI

OUTPUT_MODEL_PATH = "/artefact/model.pkl"


def compute_log_metrics(y_true, y_pred):
    """Compute and log metrics."""
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    print("Evaluation\n"
          f"  Accuracy          = {acc:.4f}\n"
          f"  F1 score          = {f1_score:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("F1 score", f1_score)


def main():
    """Entry point to perform training."""
    print("\nLoad train data")
    data = pd.read_csv("gs://bedrock-sample/otto_data/otto_data.csv")
    data["feat_1"] = (data["feat_1"].values > 0).astype(int)  # convert to binary
    print("  Train data shape:", data.shape)

    train, valid = train_test_split(data, test_size=0.2, random_state=0)
    x_train = train[FEATURES]
    y_train = train[TARGET].values
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET].values

    print("\nTrain model")
    lgb_clf = lgb.LGBMClassifier(
        num_leaves=34,
        max_depth=8,
    )
    lgb_clf.fit(x_train, y_train)

    print("\nEvaluate")
    y_pred = lgb_clf.predict(x_valid)
    compute_log_metrics(y_valid, y_pred)

    print("\nSave model")
    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(lgb_clf, model_file)

    analyzer = (
        ModelAnalyzer(lgb_clf, 'tree_model', model_type=ModelTypes.TREE)
        .test_features(x_valid)
    )
    analyzer.fairness_config(CONFIG_FAI).test_labels(y_valid).test_inference(y_pred)
    analyzer.analyze()


if __name__ == "__main__":
    main()

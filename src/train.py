import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
import shap
import joblib
import warnings

from feast import FeatureStore

warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_model():
    try:
        # --- Connect to Feast ---
        feature_repo_path = Path(__file__).parent.parent / "feature_repo"
        store = FeatureStore(repo_path=str(feature_repo_path))

        # --- Load raw CSV with target ---
        raw_data_path = Path(__file__).parent.parent / "data" / "train.csv"
        entity_df = pd.read_csv(raw_data_path)

        if "event_timestamp" not in entity_df.columns:
            entity_df["event_timestamp"] = pd.to_datetime("2025-10-08")

        if "Response" not in entity_df.columns:
            raise ValueError("CSV must contain 'Response' column")

        entity_df = entity_df[["Id", "event_timestamp", "Response"]]

        # --- Features to use in both training & API ---
        features_to_get = [
            "applicant_features:Product_Info_4",
            "applicant_features:Ins_Age",
            "applicant_features:Ht",
            "applicant_features:Wt",
            "applicant_features:BMI",
            "applicant_features:Employment_Info_1",
            "applicant_features:Employment_Info_4",
            "applicant_features:Employment_Info_6",
            "applicant_features:Insurance_History_5",
            "applicant_features:Family_Hist_2",
            "applicant_features:Family_Hist_3",
            "applicant_features:Family_Hist_4",
            "applicant_features:Family_Hist_5",
            "applicant_features:Medical_History_1",
            "applicant_features:Medical_History_10",
            "applicant_features:Medical_History_15",
            "applicant_features:Medical_History_24",
            "applicant_features:Medical_History_32",
        ]

        # --- Retrieve Historical Features ---
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=features_to_get
        ).to_df()

        training_df.fillna(0, inplace=True)

        TARGET = "Response"
        FEATURES = [col for col in training_df.columns if col not in ["Id", "event_timestamp", TARGET]]

        X = training_df[FEATURES]
        y = training_df[TARGET]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- MLflow Experiment ---
        mlflow.set_experiment("prudential-risk-prediction")
        with mlflow.start_run() as run:
            print(f"MLflow Run ID: {run.info.run_id}")

            # --- Train LightGBM ---
            params = {
                "objective": "multiclass",
                "num_class": 8,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "n_estimators": 200,
                "learning_rate": 0.03,
                "num_leaves": 20,
                "max_depth": 5,
                "seed": 42,
                "n_jobs": -1
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train - 1)

            # --- Evaluate ---
            preds = model.predict(X_val) + 1
            kappa = cohen_kappa_score(y_val, preds, weights="quadratic")
            print(f"Validation Quadratic Weighted Kappa: {kappa:.4f}")

            # --- Log model ---
            mlflow.log_params(params)
            mlflow.log_metric("quadratic_weighted_kappa", kappa)
            mlflow.lightgbm.log_model(model, "model")

            # --- Generate SHAP explainer for these 18 features only ---
            explainer = shap.TreeExplainer(model)
            explainer_dir = Path(__file__).parent.parent / "artifacts"
            explainer_dir.mkdir(exist_ok=True)
            explainer_path = explainer_dir / "shap_explainer.joblib"
            joblib.dump(explainer, explainer_path)
            mlflow.log_artifact(str(explainer_path), artifact_path="explainer")

            print("✅ Model and SHAP explainer saved and logged.")

    except Exception as e:
        print(f"Error: {e}")
        raise e


if __name__ == "__main__":
    train_model()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import joblib
import pandas as pd
from pathlib import Path
import warnings
from feast import FeatureStore

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Application Setup ---
app = FastAPI(
    title="Prudential Risk Prediction API",
    description="API for predicting insurance risk using an ML model.",
    version="1.0"
)

# --- Pydantic Models for Input Validation ---
class ApplicantInfo(BaseModel):
    Id: int

# --- Load MLflow Model and SHAP Explainer at Startup ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment_name = "prudential-risk-prediction"
df_runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time desc"])
if df_runs.empty:
    raise RuntimeError("No MLflow runs found for the experiment.")

latest_run_id = df_runs.iloc[0]['run_id']
print(f"Loading model from MLflow run ID: {latest_run_id}")

try:
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)
    print("LightGBM model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load MLflow model: {e}")

# Get feature order
model_feature_order = model.booster_.feature_name()
print(f"Model expects features in this order: {model_feature_order}")

try:
    client = mlflow.tracking.MlflowClient()
    explainer_path = client.download_artifacts(latest_run_id, "explainer/shap_explainer.joblib")
    explainer = joblib.load(explainer_path)
    print("SHAP explainer loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load SHAP explainer: {e}")

# Connect to Feast Feature Store
try:
    feature_repo_path = Path(__file__).parent.parent / "feature_repo"
    store = FeatureStore(repo_path=str(feature_repo_path))
    print("Successfully connected to Feast feature store.")
except Exception as e:
    raise RuntimeError(f"Failed to connect to Feast: {e}")


# --- API Endpoint ---
@app.post("/predict_risk")
def predict_risk(applicant_info: ApplicantInfo):
    try:
        print(f"Fetching features for applicant ID: {applicant_info.Id}")

        feature_vector_raw = store.get_online_features(
            features=[
                "applicant_features:Product_Info_4", "applicant_features:Ins_Age", "applicant_features:Ht",
                "applicant_features:Wt", "applicant_features:BMI", "applicant_features:Employment_Info_1",
                "applicant_features:Employment_Info_4", "applicant_features:Employment_Info_6",
                "applicant_features:Insurance_History_5", "applicant_features:Family_Hist_2",
                "applicant_features:Family_Hist_3", "applicant_features:Family_Hist_4",
                "applicant_features:Family_Hist_5", "applicant_features:Medical_History_1",
                "applicant_features:Medical_History_10", "applicant_features:Medical_History_15",
                "applicant_features:Medical_History_24", "applicant_features:Medical_History_32",
            ],
            entity_rows=[{"Id": applicant_info.Id}],
        ).to_df()

        if feature_vector_raw.empty:
            raise HTTPException(status_code=404, detail=f"Applicant ID {applicant_info.Id} not found in feature store.")

        feature_vector_raw.drop(columns=['Id'], inplace=True)
        feature_vector_raw.fillna(0, inplace=True)

        # Reorder features to match training order
        for feat in model_feature_order:
            if feat not in feature_vector_raw.columns:
                feature_vector_raw[feat] = 0

        feature_vector = feature_vector_raw[model_feature_order]
        # --- Prediction ---
        prediction = model.predict(feature_vector)
        risk_score = int(round(prediction[0])) + 1

        # --- SHAP Explanation ---
        shap_values = explainer.shap_values(feature_vector)
        if isinstance(shap_values, list):
            class_index = int(prediction[0])
            shap_values_for_class = shap_values[class_index][0]
        else:
            shap_values_for_class = shap_values[0]

        shap_values_for_class = shap_values_for_class.flatten()
        feature_names = list(feature_vector.columns)
        min_len = min(len(shap_values_for_class), len(feature_names))
        shap_values_for_class = shap_values_for_class[:min_len]
        feature_names = feature_names[:min_len]

        # Create SHAP DataFrame
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_for_class
        })
        shap_df['abs_shap_value'] = shap_df['shap_value'].abs()

        top_3_features = shap_df.sort_values(by='abs_shap_value', ascending=False).head(3)
        explanation = {f.replace('_', ' '): round(v, 2) for f, v in zip(top_3_features['feature'], top_3_features['shap_value'])}

        return {
            "applicant_id": applicant_info.Id,
            "predicted_risk_level": risk_score,
            "explanation": {
                "message": "Top 3 factors influencing the prediction (positive values increase risk).",
                "factors": explanation
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

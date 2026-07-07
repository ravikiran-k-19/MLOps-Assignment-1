"""
Pydantic request / response schemas for the prediction API.

PatientFeatures  → POST /predict request body
PredictionResponse → POST /predict response body
"""

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(..., description="Age in years", examples=[63])
    sex: float = Field(..., description="Sex (1=male, 0=female)", examples=[1])
    cp: float = Field(..., description="Chest pain type: 0=asymptomatic, 1=atypical angina, 2=non-anginal, 3=typical angina", examples=[3])
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg on admission)", examples=[145])
    chol: float = Field(..., description="Serum cholesterol (mg/dl)", examples=[233])
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)", examples=[1])
    restecg: float = Field(..., description="Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)", examples=[0])
    thalach: float = Field(..., description="Maximum heart rate achieved", examples=[150])
    exang: float = Field(..., description="Exercise induced angina (1=yes, 0=no)", examples=[0])
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest", examples=[2.3])
    slope: float = Field(..., description="Slope of peak exercise ST segment (0=downsloping, 1=flat, 2=upsloping)", examples=[0])
    ca: float = Field(..., description="Number of major vessels coloured by fluoroscopy (0-3)", examples=[0])
    thal: float = Field(..., description="Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)", examples=[1])

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Heart Disease, 1 = Heart Disease")
    confidence: float = Field(..., description="Predicted probability of heart disease (0.0–1.0)")
    model_version: str = Field(..., description="Deployed model version identifier")

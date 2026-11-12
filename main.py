import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import joblib
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(APP_DIR / "models" / "iris_random_forest_model.joblib"))
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:7001")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "iris-classifier")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def load_model():
    """Load the bundled model by default, or an MLflow model when configured."""
    if MLFLOW_MODEL_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
    return joblib.load(MODEL_PATH)


def load_model_version():
    if MLFLOW_MODEL_URI:
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_MODEL_ALIAS,
        )
        return str(version.version)
    return os.getenv("MODEL_VERSION", "local")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

def create_app(
    model_loader: Callable = load_model,
    version_loader: Callable | None = None,
):
    if version_loader is None:
        version_loader = load_model_version

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(
                f"Failed to load model: {ex}"
            )

        app.state.model_version = version_loader()
        yield

    app = FastAPI(
        title="Iris Classification API",
        description="Iris prediction API powered by MLFlow",
        version="3.1.0",
        lifespan=lifespan
    )
        
    @app.get("/")
    def ready_to_serve():
        return {"status": "Ready to serve"}

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_name": "iris-classification",
            "model_version": app.state.model_version
            }

    # Endpoint for making predictions with the Iris Random Forest model
    @app.post("/predict")
    def predict(data: IrisInput, request: Request):
        model = request.app.state.model
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]
        prediction = int(model.predict(features)[0])
        print("Prediction is", prediction)
        return {
            "prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction]
        }

    return app


# Production application instance
app = create_app()

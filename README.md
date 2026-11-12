# Iris ML API

FastAPI service for Iris flower classification. The `/predict` endpoint accepts sepal and petal measurements and returns a predicted class. The application loads the `iris-classifier` model's `champion` alias from the configured MLflow tracking server when it starts.

## Run locally

1. Install dependencies with `pip install -r requirements.txt`.
2. Set `MLFLOW_TRACKING_URI` to a running MLflow tracking server that has the registered `iris-classifier` model and its `champion` alias.
3. Start the API with `uvicorn main:app --host 0.0.0.0 --port 8000`.

The API documentation is available at `/docs`; health and prediction routes are `/health` and `/predict`.

## Docker

Build the image with `docker build -t iris-ml-api .`. Run it with the MLflow tracking server reachable from the container and set `MLFLOW_TRACKING_URI` to that server's URL.

## CI pipeline

The GitHub Actions workflow runs the API tests on pushes and pull requests targeting `main`. On pushes to `main`, it also builds and publishes the Docker image to GitHub Container Registry (GHCR).

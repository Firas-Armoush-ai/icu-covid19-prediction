import os
# purpose: prepare for deployment, switch environments WITHOUT modifying code.

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"  # FastAPI runs in the same container on Render and locally
)

PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
MODEL_INFO_ENDPOINT = f"{API_BASE_URL}/model-info"

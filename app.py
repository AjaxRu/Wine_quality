from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from Wine_quality_model.config.core import config, TRAINED_MODEL_DIR
from Wine_quality_model.processing.data_manager import load_pipeline
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Настройка шаблонов Jinja2
templates = Jinja2Templates(directory="templates")


class WineFeatures(BaseModel):
    features: List[float] = Field(..., min_items=11, max_items=11)


@app.on_event("startup")
def load_model():
    global model
    model_path = TRAINED_MODEL_DIR / (config.app_config.pipeline_save_file + ".pkl")
    model = load_pipeline(file_name=model_path)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, features: str = Form(...)):
    try:
        feature_list = [float(x) for x in features.split(',')]
        if len(feature_list) != 11:
            raise ValueError("Incorrect number of features. Expected 11.")

        input_data = pd.DataFrame([feature_list], columns=config.model_config_params.features)
        predictions = model.predict(input_data)
        return templates.TemplateResponse("result.html", {"request": request, "prediction": predictions[0]})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)




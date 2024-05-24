import sys
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel, Field
from typing import List
from Wine_quality_model.config.core import config, TRAINED_MODEL_DIR
from Wine_quality_model.processing.data_manager import load_pipeline


class InputDataSchema(BaseModel):
    features: List[float] = Field(..., min_items=11, max_items=11)


def make_prediction(input_data: List) -> np.ndarray:
    model_path = TRAINED_MODEL_DIR / (config.app_config.pipeline_save_file + ".pkl")
    model = load_pipeline(file_name=model_path)
    data = pd.DataFrame(input_data, columns=config.model_config_params.features)
    predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    # Example: python predict.py 7.4 0.7 0.0 1.9 0.076 11.0 34.0 0.9978 3.51 0.56 9.4
    input_features = [float(value) for value in sys.argv[1:]]
    if len(input_features) != 11:
        print(f"Expected 11 input features, but got {len(input_features)}")
        sys.exit(1)

    input_data = [input_features]
    prediction = make_prediction(input_data=input_data)
    print(f"Prediction: {prediction}")






















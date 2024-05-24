import pandas as pd
from pydantic import BaseModel, ValidationError, validator
from typing import List

class DataSchema(BaseModel):
    fixed_acidity: List[float]
    volatile_acidity: List[float]
    citric_acid: List[float]
    residual_sugar: List[float]
    chlorides: List[float]
    free_sulfur_dioxide: List[float]
    total_sulfur_dioxide: List[float]
    density: List[float]
    pH: List[float]
    sulphates: List[float]
    alcohol: List[float]
    quality: List[int]

    @validator('*')
    def check_na(cls, v):
        if pd.isna(v).any():
            raise ValueError("Missing values detected")
        return v

def validate_dataset(df: pd.DataFrame) -> None:
    try:
        DataSchema(**df.to_dict(orient="list"))
    except ValidationError as e:
        raise e

def check_na(df: pd.DataFrame) -> None:
    if df.isna().sum().sum() > 0:
        raise ValueError("Data contains NA values. Please clean the data and try again.")



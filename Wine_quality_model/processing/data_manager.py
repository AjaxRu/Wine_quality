import pandas as pd
from pathlib import Path
import joblib
from Wine_quality_model.config.core import config, TRAINED_MODEL_DIR

def load_dataset(file_path: str) -> pd.DataFrame:
    dataset_path = Path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"File not found: {dataset_path}")
    return pd.read_csv(dataset_path)

def save_pipeline(*, pipeline_to_persist, file_name: str) -> None:
    """Persist the pipeline."""
    save_file_name = f"{file_name}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)

def load_pipeline(*, file_name: str):
    """Load a persisted pipeline."""
    file_path = TRAINED_MODEL_DIR / file_name
    return joblib.load(file_path)

def remove_old_pipelines(*, files_to_keep: list) -> None:
    """Remove old model pipelines."""
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()














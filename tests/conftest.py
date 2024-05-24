import pytest
from Wine_quality_model.config.core import config
from Wine_quality_model.processing.data_manager import load_dataset

print("conftest.py is being loaded")

@pytest.fixture()
def sample_input_data():
    file_path = config.app_config.test_data_file
    print(f"Loading test data from: {file_path}")
    return load_dataset(file_path=file_path)






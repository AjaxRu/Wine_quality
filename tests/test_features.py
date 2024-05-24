import pandas as pd
from Wine_quality_model.config.core import config


def test_dataset_loading(sample_input_data):
    # Given
    data = sample_input_data

    # Then
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert list(data.columns) == config.model_config_params.features + [config.model_config_params.target]






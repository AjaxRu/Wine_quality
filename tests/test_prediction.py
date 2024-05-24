import numpy as np
import pytest
from Wine_quality_model.predict import make_prediction
from Wine_quality_model.config.core import config

def test_make_prediction(sample_input_data):
    sample = sample_input_data.sample(1)
    input_data = [list(sample[config.model_config_params.features].values[0])]

    result = make_prediction(input_data=input_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert isinstance(result[0], (np.int64, np.int32, int))








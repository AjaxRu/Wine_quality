import pytest
from Wine_quality_model.config.core import config

def test_model_config_params():
    # Проверяем наличие и типы основных параметров конфигурации
    assert isinstance(config.model_config_params.target, str), "target should be a string"
    assert isinstance(config.model_config_params.features, list), "features should be a list"
    assert all(isinstance(feature, str) for feature in config.model_config_params.features), "all features should be strings"
    assert isinstance(config.model_config_params.test_size, float), "test_size should be a float"
    assert isinstance(config.model_config_params.random_state, int), "random_state should be an integer"
    assert isinstance(config.model_config_params.n_estimators, list), "n_estimators should be a list"
    assert all(isinstance(n, int) for n in config.model_config_params.n_estimators), "all n_estimators should be integers"
    assert isinstance(config.model_config_params.max_depth, list), "max_depth should be a list"
    assert all(isinstance(depth, int) for depth in config.model_config_params.max_depth), "all max_depth should be integers"

def test_app_config_params():
    # Проверяем наличие и типы параметров конфигурации приложения
    assert isinstance(config.app_config.package_name, str), "package_name should be a string"
    assert isinstance(config.app_config.training_data_file, str), "training_data_file should be a string"
    assert isinstance(config.app_config.test_data_file, str), "test_data_file should be a string"
    assert isinstance(config.app_config.pipeline_save_file, str), "pipeline_save_file should be a string"

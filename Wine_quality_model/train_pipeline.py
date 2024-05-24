import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from Wine_quality_model.config.core import config, TRAINED_MODEL_DIR
from Wine_quality_model.processing.data_manager import load_dataset, save_pipeline
from Wine_quality_model.processing.validate import validate_dataset, check_na
from Wine_quality_model.pipeline import pipeline

# Игнорирование конкретного предупреждения
warnings.filterwarnings("ignore", message='Field "model_config_params" has conflict with protected namespace "model_".')

def run_training():
    print("Configuration loaded successfully.")
    print(f"Train data path: {config.app_config.training_data_file}")
    print(f"Test data path: {config.app_config.test_data_file}")

    train_df = load_dataset(config.app_config.training_data_file)
    test_df = load_dataset(config.app_config.test_data_file)

    # Validate dataset
    try:
        check_na(train_df)
        check_na(test_df)
        validate_dataset(train_df)
        validate_dataset(test_df)
    except ValueError as e:
        print(e)
        return

    X_train = train_df[config.model_config_params.features]
    y_train = train_df[config.model_config_params.target]
    X_test = test_df[config.model_config_params.features]
    y_test = test_df[config.model_config_params.target]

    # Hyperparameter tuning using GridSearchCV with StratifiedKFold
    parameters = {
        'classifier__n_estimators': config.model_config_params.n_estimators,
        'classifier__max_depth': config.model_config_params.max_depth
    }
    stratified_kfold = StratifiedKFold(n_splits=3)  # Use 3-fold stratified cross-validation
    clf = GridSearchCV(pipeline, parameters, cv=stratified_kfold)

    print("Searching for best hyperparameters ...")
    clf.fit(X_train, y_train)
    print(f'Best Hyperparameters: {clf.best_params_}')

    # Train the model with the best hyperparameters
    best_pipeline = clf.best_estimator_
    best_pipeline.fit(X_train, y_train)

    # Evaluate model on training data
    train_predictions = best_pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions, average='weighted')

    print(f"Training Accuracy: {train_accuracy}")
    print(f"Training F1 Score: {train_f1}")

    # Evaluate model on test data
    test_predictions = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')

    print(f"Testing Accuracy: {test_accuracy}")
    print(f"Testing F1 Score: {test_f1}")
    print('\n')
    print(classification_report(y_test, test_predictions, zero_division=1))

    # Confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the best model
    save_pipeline(pipeline_to_persist=best_pipeline, file_name=config.app_config.pipeline_save_file)
    print(f"Model saved to {TRAINED_MODEL_DIR / (config.app_config.pipeline_save_file + '.pkl')}")

if __name__ == '__main__':
    run_training()






























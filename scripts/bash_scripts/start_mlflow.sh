#!/bin/bash
# Создаем директорию для MLflow данных, если её нет
mkdir -p mlflow_data/backend
mkdir -p mlflow_data/artifacts

echo "Запускаем MLflow server..."
mlflow server \
  --backend-store-uri sqlite:///mlflow_data/backend/mlflow.db \
  --default-artifact-root ./mlflow_data/artifacts \
  --host 0.0.0.0 \
  --port 5000
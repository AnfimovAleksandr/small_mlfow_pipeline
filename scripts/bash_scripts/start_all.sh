#!/bin/bash
set -e

# Создаем директорию для MLflow данных, если её нет
mkdir -p mlflow_data/backend
mkdir -p mlflow_data/artifacts

# Запускаем MLflow server в фоновом режиме
echo "Запускаем MLflow server..."
mlflow server \
  --backend-store-uri sqlite:///mlflow_data/backend/mlflow.db \
  --default-artifact-root ./mlflow_data/artifacts \
  --host 0.0.0.0 \
  --port 5000 &
MLFLOW_PID=$!

# Ждем, пока MLflow станет доступен
echo "Ожидаем доступности MLflow..."
until curl -s http://localhost:5000/health >/dev/null; do
  echo -n "."
  sleep 1
done
echo -e "\nMLflow доступен!"

# Запускаем наше приложение
echo "Запускаем приложение..."
python app.py

# При завершении останавливаем MLflow
trap "kill $MLFLOW_PID" EXIT
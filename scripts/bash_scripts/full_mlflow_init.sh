#!/bin/bash
set -e

# Запускаем MLflow server в фоновом режиме
echo "Запускаем MLflow server..."
mlflow server \
  --backend-store-uri sqlite:////mlflow_app/mlflow_data/backend/mlflow.db \
  --default-artifact-root /mlflow_app/mlflow_data/artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts &
MLFLOW_PID=$!

# Ждем, пока MLflow станет доступен
echo "Ожидание доступности MLflow..."
until curl -s http://localhost:5000/health >/dev/null; do
  echo -n "."
  sleep 1
done
echo -e "\nMLflow доступен!"

# Обучаем модель и продвигаем ее в Production
echo "Запускаем обучение модели и продвижение в Production..."
python ./scripts/train_logreg_model.py
python ./scripts/train_lgbm_model.py
python ./scripts/promote_best_model.py

# Оставляем MLflow server запущенным
wait $MLFLOW_PID
# Dockerfile.mlflow
FROM python:3.10

# Нужно для корректного запуска mlflow (чтобы проверить health)
RUN apt-get update && apt-get install -y curl

WORKDIR /mlflow_app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY data/processed data/processed
COPY scripts scripts
COPY models/preprocessor.pkl models/

RUN chmod +x scripts/bash_scripts/full_mlflow_init.sh

# Создаем директорию для данных MLflow и устанавливаем права
RUN mkdir -p ./mlflow_data/backend && \
    mkdir -p ./mlflow_data/artifacts && \
    chmod -R 777 ./mlflow_data

CMD ["scripts/bash_scripts/full_mlflow_init.sh"]
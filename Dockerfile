FROM python:3.10-slim

# Нужно для корректного запуска mlflow (чтобы проверить health)
RUN apt-get update && apt-get install -y curl

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x scripts/bash_scripts/start_all.sh

VOLUME ["/app/mlflow_data"]

EXPOSE 8000

CMD ["/app/scripts/bash_scripts/start_all.sh"]
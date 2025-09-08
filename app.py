from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import datetime
import mlflow
from mlflow.tracking import MlflowClient

import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/scripts')

from scripts.data_preprocessor_class import KickstarterPreprocessor
from sklearn.metrics import (accuracy_score, f1_score, average_precision_score)

app = Flask(__name__)

# Глобальные переменные для отслеживания состояния
model = None
preprocessor = None
model_framework = None
model_loaded = False

# Загружаем компоненты при старте
mlflow.set_tracking_uri('http://localhost:5000')

def load_production_model():
    """Загружает Production модель и связанный с ней предобработчик"""
    global model, preprocessor, model_loaded, model_framework
    
    try:
        # Устанавливаем tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        print(mlflow_uri)
        print(os.getcwd())

        # client = MlflowClient()
        
        # # Ищем Production версию модели
        # prod_versions = client.get_latest_versions("KickstarterModel", stages=["Production"])
        
        # if not prod_versions:
        #     print("Нет модели в статусе Production")
        #     model_loaded = False
        #     return False
        
        # prod_version = prod_versions[0]
        # print(f"Найдена Production версия: {prod_version.version}")
        
        # # Получаем инфу о фреймворке логгирования для корректной загрузки
        # run_id = prod_version.run_id
        # run_data = client.get_run(run_id)
        # model_type = run_data.data.tags.get("model_type", "unknown")
        # model_framework = run_data.data.tags.get("model_framework", "unknown")
        # print(f"Информация о модели: тип={model_type}, фреймворк={model_framework}")

        # # Загружаем саму модель

        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     model_local_path = mlflow.artifacts.download_artifacts(
        #             run_id=run_id,
        #             artifact_path="logreg_model",
        #             dst_path=tmp_dir
        #         )
            
        #     if model_framework == "sklearn":
        #         model = mlflow.sklearn.load_model(model_local_path)
        #     elif model_framework == "lightgbm":   
        #         model = mlflow.lightgbm.load_model(model_local_path)
        #     else:
        #         print("Ошибка: Неизвестный тип модели")
        #         model_loaded = False
        #         return False
        model_framework = "sklearn"
        model = joblib.load('/app/models/logreg_model.pkl')
        print('Модель загружена')
        
        
        # # Создаем временный каталог для загрузки артефакта
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     preprocessor_local_path = mlflow.artifacts.download_artifacts(
        #             run_id=run_id,
        #             artifact_path="preprocessor/preprocessor.pkl",
        #             dst_path=tmp_dir
        #         )
        #     print(f"Предобработчик загружен в: {preprocessor_local_path}")
            
        #     # Загружаем предобработчик
        #     preprocessor = joblib.load(preprocessor_local_path)

        preprocessor = joblib.load('/app/models/preprocessor.pkl')

        model_loaded = True
        print("Модель и предобработчик успешно загружены")
        return True
    
    except Exception as e:
        print(f"Ошибка загрузки: {str(e)}")
        model_loaded = False
        return False
    
load_production_model()

@app.route('/health', methods=['GET'])
def health():
    """Базовая проверка работоспособности сервиса (всегда возвращает 200)"""
    return jsonify({
        'status': 'healthy',
        'service': 'Kickstarter Prediction API',
        'timestamp': datetime.datetime.now().isoformat()
    }), 200

@app.route('/ready', methods=['GET'])
def ready():
    """Проверка готовности к обработке запросов (проверяет загрузку моделей)"""
    if not model_loaded:
        return jsonify({
            'status': 'unhealthy',
            'reason': 'Model or preprocessor not loaded',
            'timestamp': datetime.datetime.now().isoformat()
        }), 503  # Service Unavailable
    
    return jsonify({
        'status': 'ready',
        'service': 'Kickstarter Prediction API',
        'model_type': 'LogisticRegression',  
        'timestamp': datetime.datetime.now().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():

    # Доступность
    if not model_loaded:
        return jsonify({
            "error": "Service is not ready. Models are still loading."
        }), 503
    
    # Валидация входных данных
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    input_data = request.json
    
    required_fields = ['project_id','name','desc','goal','keywords','disable_communication','country','currency','deadline','state_changed_at','created_at','launched_at','backers_count']
    if not all(field in input_data for field in required_fields):
        return jsonify({
            "error": "Missing required fields",
            "required": required_fields
        }), 400
    
    try:
        # Обработка
        df = pd.DataFrame([input_data])
        data = preprocessor.preprocess_raw_before_scaling(df, testing = True)
        processed_data = preprocessor.transform(data)
        
        # Предсказание
        if model_framework == "sklearn":
            prediction = model.predict(processed_data)
            probabilities = model.predict_proba(processed_data)
        elif model_framework == "lightgbm":
            probabilities = model.predict(processed_data)
            prediction = (probabilities >= 0.5).astype(int)[0]
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": probabilities[0].tolist(),
            "classes": model.classes_.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
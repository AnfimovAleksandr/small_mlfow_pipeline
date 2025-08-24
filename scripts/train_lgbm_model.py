import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import json
from time import time

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, average_precision_score)

from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

# Здесь будет ссылка на оригинальный гитхаб с полным проектом

def process_result(model, X_train, X_test, y_train, y_test):
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Кросс-валидация на обучающей выборке
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return {
        'f1_test': f1_score(y_test, y_pred),
        'PR_AUC_test': average_precision_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_cv_mean': cv_scores.mean(),
        'f1_cv_std': cv_scores.std(),
        }

def main():
    # Подгружаем train
    train_data = pd.read_csv('data/train.csv')
    X_train = train_data.drop(['final_status'], axis=1)
    y_train = train_data['final_status']

    # Подгружаем test
    test_data = pd.read_csv('data/test.csv')
    X_test = test_data.drop(['final_status'], axis=1)
    y_test = test_data['final_status']

    mlflow.set_experiment('kaggle_projects')

    with mlflow.start_run(run_name="LGBM_optimal"):
        # ---------------------------- LightGBM ----------------------------

        # Параметры
        best_params = {
            'n_estimators': 493, 
            'max_depth': 3, 
            'learning_rate': 0.10426716049785455, 
            'feature_fraction': 0.9947089259050176, 
            'bagging_fraction': 0.8148654260211938, 
            'min_child_samples': 9, 
            'reg_alpha': 5.5112247353974215, 
            'reg_lambda': 9.561295790261457,
            "is_unbalance": True, # Для дисбаланса классов
            'random_state': 42,
            'verbosity': -1,
            'force_col_wise': True
        }

        # Обучаем сразу оптимальную (подбор см в основном проекте)
        print('Начаинаем обучение LightGBM бустинга')
        start = time()
        lgbm_model = lgb.LGBMClassifier(**best_params)
        lgbm_model.fit(X_train, y_train)
        end = time()
        print('Обучение завершено')

        # Логируем параметры
        print('Логирование параметров')
        mlflow.log_params(best_params)

        # Подсчитываем метрики
        print('Считаем метрики')
        results = process_result(lgbm_model, X_train, X_test, y_train, y_test)

        # Логируем метрики
        print('Логируем метрики')
        mlflow.log_metrics(results)

        # Дополнительные данные
        print('Логируем допданные')
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))
        mlflow.log_param('training_time_sec', round(end - start, 2))


        # Логируем модель
        print('Логируем модель')

        input_example = X_test[:1]
        signature = mlflow.models.infer_signature(X_test[:2], lgbm_model.predict(X_test[:2]))

        mlflow.lightgbm.log_model(
            lgb_model=lgbm_model,
            name="lgbm_model",
            signature=signature,
            input_example=input_example
        )

        with open('models/lgbm_model.pkl', 'wb') as f:
            pickle.dump(lgbm_model, f)

        print("Pipeline для бустинга LightGBM завершён")
        
main()

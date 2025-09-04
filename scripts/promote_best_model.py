import mlflow

from mlflow.tracking import MlflowClient

def promote_best_model(model_name):
    print()
    print('Сравниваем модели')
    mlflow.set_tracking_uri('http://localhost:5000')
    client = MlflowClient()
    best_f1_score = 0
    best_version = None
    for version in client.search_model_versions(f"name='{model_name}'"):
        tmp_f1_score = version.tags.get("f1_score")
        if tmp_f1_score:
            tmp_f1_score = float(tmp_f1_score)
            if tmp_f1_score > best_f1_score:
                best_f1_score = tmp_f1_score
                best_version = version

    if best_version:
        print(f'Лучшая версия {best_version}')
        client.transition_model_version_stage(
            name=best_version.name,
            version=best_version.version,
            stage="Production"
        )
    print('Promotion complete!')

promote_best_model('KickstarterModel')
from airflow.decorators import dag, task
from datetime import datetime

# Определяем DAG
@dag(
    dag_id="manual_testing_run",
    description="DAG для ручного тестирования",
    start_date=datetime(2025, 1, 1),  
    schedule=None,          
    catchup=False
)

def ml_pipeline():

    @task.bash
    def pull_dvc():
        return 'dvc pull'
    
    @task.bash
    def load_data():
        return "python scripts/data_loader.py"  

    @task.bash
    def train_logreg_model():
        return "python scripts/train_logreg_model.py"

    @task.bash
    def train_lgbm_model():
        return "python scripts/train_lgbm_model.py"

    @task.bash
    def push_dvc():
        return 'dvc push'
    
    dvc_pull_step = pull_dvc()
    load_step = load_data()
    logreg_step = train_logreg_model()
    lgbm_step = train_lgbm_model()
    dvc_push_step = push_dvc()


    dvc_pull_step >> load_step >> [logreg_step, lgbm_step] >> dvc_push_step

ml_pipeline()

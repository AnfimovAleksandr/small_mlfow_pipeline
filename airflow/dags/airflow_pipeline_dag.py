from airflow.decorators import dag, task
from datetime import datetime, timedelta


deafult_args = {
    'owner': 'anfimov_aleksandr',
    'depends_on_past': False,
    'start_date': datetime(2000, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes = 5)
}

# Определяем DAG
@dag(
    dag_id="manual_testing_run",
    deafult_args = deafult_args,
    description="DAG для ручного тестирования", 
    schedule=None,          
    catchup=False,
    tags=['ml', 'dvc']
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

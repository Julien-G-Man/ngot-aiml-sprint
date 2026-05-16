# Run with: poetry run python scripts/first_mlflow_run.py

import mlflow

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('my-first-experiment')

with mlflow.start_run(run_name='learning-mlflow') as run:

    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('learning_rate', 0.05)
    mlflow.log_param('model_type', 'gradient_boosting')
    mlflow.log_params({'max_depth': 4, 'subsample': 0.8})

    mlflow.log_metric('train_mae', 22.4)
    mlflow.log_metric('val_mae', 24.1)
    mlflow.log_metric('val_r2', 0.89)

    # Log a metric at multiple steps (useful for training loss curves)
    for epoch in range(5):
        mlflow.log_metric('train_loss', 100 / (epoch + 1), step=epoch)

    with open('tmp/notes.txt', 'w') as f:
        f.write('This is my first MLflow run\n')
        f.write('Model type: Gradient Boosting\n')
    mlflow.log_artifact('tmp/notes.txt')

    mlflow.set_tags({
        'dataset': 'logistics-ghana-v1',
        'engineer': 'Julien',
        'status': 'experiment',
    })

    print(f'Run ID: {run.info.run_id}')
    print(f'Experiment ID: {run.info.experiment_id}')
    print(f'View at: http://localhost:5000/#/experiments/{run.info.experiment_id}')

# Run with: poetry run python scripts/first_mlflow_run.py

import mlflow

# Step 1: Tell MLflow where the tracking server is
mlflow.set_tracking_uri('http://localhost:5000')

# Step 2: Set (or create) an experiment
# All runs in this script will go into this experiment
mlflow.set_experiment('my-first-experiment')

# Step 3: Start a run
# Everything inside the 'with' block is logged to this single run
with mlflow.start_run(run_name='learning-mlflow') as run:

    # Log parameters — settings you chose before training
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('learning_rate', 0.05)
    mlflow.log_param('model_type', 'gradient_boosting')

    # You can also log a whole dictionary at once
    mlflow.log_params({'max_depth': 4, 'subsample': 0.8})

    # Log metrics — results measured after training
    mlflow.log_metric('train_mae', 22.4)
    mlflow.log_metric('val_mae', 24.1)
    mlflow.log_metric('val_r2', 0.89)

    # Log a metric at multiple steps (useful for training loss curves)
    for epoch in range(5):
        mlflow.log_metric('train_loss', 100 / (epoch + 1), step=epoch)

    # Log a file as an artifact
    # First, create a simple text file
    with open('tmp/notes.txt', 'w') as f:
        f.write('This is my first MLflow run\n')
        f.write('Model type: Gradient Boosting\n')
    mlflow.log_artifact('tmp/notes.txt')

    # Log tags — free-form labels for filtering
    mlflow.set_tags({
        'dataset': 'logistics-ghana-v1',
        'engineer': 'Julien',
        'status': 'experiment',
    })

    # Print the run ID so we can find it in the UI
    print(f'Run ID: {run.info.run_id}')
    print(f'Experiment ID: {run.info.experiment_id}')
    print(f'View at: http://localhost:5000/#/experiments/{run.info.experiment_id}')

import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

experiment_name = "Homework_#2"
mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog(log_datasets=False)

mlflow.sklearn.autolog()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):
    train_data_path = os.path.join(data_path, "train.pkl")
    valid_data_path = os.path.join(data_path, "val.pkl")

    X_train, y_train = load_pickle(train_data_path)
    X_val, y_val = load_pickle(valid_data_path)

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)

if __name__ == '__main__':
    run_train()

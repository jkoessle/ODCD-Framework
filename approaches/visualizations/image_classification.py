import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils

from tbparse import SummaryReader
from datetime import datetime


def get_binary_accuracy_data(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Extract binary accuracy data from Tensorboard logs.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Binary accuracy logs
    """
    filter = "epoch_binary_accuracy"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(
        columns={"value": "binary_accuracy", "dir_name": "model"})
    return tb_logs


def get_loss_data(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Extract loss data from Tensorboard logs.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame:Loss logs
    """
    filter = "epoch_loss"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(columns={"value": "loss", "dir_name": "model"})
    return tb_logs


def plot_figures(tb_logs: pd.DataFrame, output_dir: str):
    """Plot figures for binary accuracy and loss. Saves plot at output_dir.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs
        output_dir (str): Directory where to save plot
    """
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    accuracy_data = get_binary_accuracy_data(tb_logs)
    sns.lineplot(data=accuracy_data, x="step", y="binary_accuracy",
                 hue=accuracy_data["model"]).set_title("binary accuracy")
    plt.subplot(1, 2, 2)
    loss_data = get_loss_data(tb_logs)
    sns.lineplot(data=loss_data, x="step", y="loss",
                 hue=loss_data["model"]).set_title("loss")
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))


def get_binary_accuracy_data_per_run(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Get binary accuracy data per run.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Tensorboard logs filtered on accuracy data 
    """
    filter = "epoch_binary_accuracy"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(columns={"value": "binary_accuracy"})
    return tb_logs


def get_loss_data_per_run(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Get loss data per run.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Tensorboard logs filtered on loss data 
    """
    filter = "epoch_loss"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(columns={"value": "loss"})
    return tb_logs


def get_maximum_per_run(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Get maximum accuracy per run.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Tensorboard logs grouped by maximum accuracy per run 
    """
    tb_logs = get_binary_accuracy_data_per_run(tb_logs)
    return tb_logs[["dir_name", "binary_accuracy"]].groupby("dir_name").transform("max")


def get_maximum_average(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Get average maximum accuracy per model.
    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Tensorboard logs grouped by average maximum of accuracy per model 
    """
    tb_logs = get_binary_accuracy_data(tb_logs)
    return tb_logs[["model", "maximums"]].groupby("model").mean()


def get_minimal_loss_per_run(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Get minimal loss per run.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Tensorboard logs grouped by minimal loss per run 
    """
    tb_logs = get_loss_data_per_run(tb_logs)
    return tb_logs[["dir_name", "loss"]].groupby(["dir_name"]).transform("min")


def get_minimal_loss_average(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Get average minimal loss per model.
    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Tensorboard logs grouped by average loss per model 
    """
    tb_logs = get_loss_data(tb_logs)
    return tb_logs[["model", "minimals"]].groupby(["model"]).mean()


def convert_wall_time(wall_time: float) -> datetime:
    """Convert wall timestamp to datetime.

    Args:
        wall_time (float): Timestamp

    Returns:
        datetime: Datetime object
    """
    return datetime.fromtimestamp(wall_time)


def wall_time_2_datetime(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame column containing wall times to datetime format.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: DataFrame
    """
    tb_logs["wall_time"] = tb_logs["wall_time"].map(
        lambda x: convert_wall_time(x))
    return tb_logs


def convert_to_relative_time(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps to relative time.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: DataFrame converted to relative time
    """
    start = tb_logs[["wall_time", "dir_name"]].groupby(
        ["dir_name"]).transform('min')
    end = tb_logs[["wall_time", "dir_name"]].groupby(
        ["dir_name"]).transform('max')
    tb_logs["time_delta"] = end["wall_time"] - start["wall_time"]
    return tb_logs


def get_average_time(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Get average relative time per model.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: DataFrame with average relative time per model
    """
    tb_logs = get_loss_data(tb_logs)
    return tb_logs[["model", "time_delta"]].groupby(["model"]).mean()


if __name__ == "__main__":
    log_dir = "Specify tensorboard directory"
    output_dir = "Specify output directory"
    reader = SummaryReader(log_dir, extra_columns={"dir_name", "wall_time"})
    logs_df = reader.tensors
    columns = ["epoch_binary_accuracy", "epoch_loss"]
    logs_df = utils.filter_validation_logs(logs_df, columns)
    logs_df = utils.dir_name_2_model_name(logs_df)

    plot_figures(logs_df, output_dir)

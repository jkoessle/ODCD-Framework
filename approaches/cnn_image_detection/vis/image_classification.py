import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tbparse import SummaryReader


def get_binary_accuracy_data(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Extract binary accuracy data from Tensorboard logs.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Binary accuracy logs
    """
    filter = "epoch_binary_accuracy"
    tb_logs = filter_on_tag(tb_logs, filter)
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
    tb_logs = filter_on_tag(tb_logs, filter)
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


def filter_validation_logs(tb_logs: pd.DataFrame, columns) -> pd.DataFrame:
    """Filters Tensorboard logs by validation logs and selected columns only.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs
        columns (_type_): Columns to filter for

    Returns:
        pd.DataFrame: Filterd Tensorboard logs
    """
    tb_logs = tb_logs[tb_logs.dir_name.str.endswith("/validation")]
    tb_logs = tb_logs.loc[tb_logs["tag"].isin(columns)]
    return tb_logs


def filter_on_tag(tb_logs: pd.DataFrame, filter: str) -> pd.DataFrame:
    """Filters DataFrame on condition.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs
        filter (str): Filter condition

    Returns:
        pd.DataFrame: Filtered DateFrame
    """
    return tb_logs.loc[tb_logs["tag"] == filter]


def split_dir_name(dir_name: str) -> str:
    """Splits directory name and returns model name.

    Args:
        dir_name (str): Directory where Tensorboard logs are stored

    Returns:
        str: Model name
    """
    split = dir_name.split("_")
    if len(split) == 4:
        return split[-1].split("/")[0]
    elif len(split) == 5:
        return f"{split[-2]}_{split[-1]}".split("/")[0]


def dir_name_2_model_name(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Updates directory name to model name.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Updated Tensorboard logs
    """
    tb_logs["dir_name"] = tb_logs["dir_name"].map(lambda x: split_dir_name(x))
    return tb_logs


if __name__ == "__main__":
    log_dir = "Specify tensorboard directory"
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    logs_df = reader.tensors
    columns = ["epoch_binary_accuracy", "epoch_loss"]
    logs_df = filter_validation_logs(logs_df, columns)
    logs_df = dir_name_2_model_name(logs_df)

    plot_figures(logs_df)

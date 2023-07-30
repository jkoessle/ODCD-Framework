import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils

from tbparse import SummaryReader


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


if __name__ == "__main__":
    log_dir = "Specify tensorboard directory"
    output_dir = "Specify output directory"
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    logs_df = reader.tensors
    columns = ["epoch_binary_accuracy", "epoch_loss"]
    logs_df = utils.filter_validation_logs(logs_df, columns)
    logs_df = utils.dir_name_2_model_name(logs_df)

    plot_figures(logs_df, output_dir)

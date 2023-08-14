import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils

from tbparse import SummaryReader


def get_best_ap_data(tb_logs: pd.DataFrame) -> pd.DataFrame:
    filter = "best_AP"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(
        columns={"value": "best_AP", "dir_name": "model"})
    return tb_logs


def get_ap_data(tb_logs: pd.DataFrame) -> pd.DataFrame:
    filter = "AP"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(
        columns={"value": "AP", "dir_name": "model"})
    return tb_logs


def get_total_loss_data(tb_logs: pd.DataFrame) -> pd.DataFrame:
    filter = "total_loss"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(
        columns={"value": "total_loss", "dir_name": "model"})
    return tb_logs


def get_box_loss_data(tb_logs: pd.DataFrame) -> pd.DataFrame:
    filter = "box_loss"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(
        columns={"value": "box_loss", "dir_name": "model"})
    return tb_logs


def get_cls_loss_data(tb_logs: pd.DataFrame) -> pd.DataFrame:
    filter = "cls_loss"
    tb_logs = utils.filter_on_tag(tb_logs, filter)
    tb_logs = tb_logs.rename(
        columns={"value": "cls_loss", "dir_name": "model"})
    return tb_logs


def split_dir_name(dir_name: str) -> str:
    """Splits directory name and returns model name.

    Args:
        dir_name (str): Directory where Tensorboard logs are stored

    Returns:
        str: Model name
    """
    split = dir_name.split("_")
    if len(split) == 3:
        return f"{split[-2]}_{split[-1]}".split("/")[0]
    elif len(split) == 4:
        return f"{split[-3]}_{split[-2]}_{split[-1]}".split("/")[0]


def dir_name_2_model_name(tb_logs: pd.DataFrame) -> pd.DataFrame:
    """Updates directory name to model name.

    Args:
        tb_logs (pd.DataFrame): Tensorboard logs

    Returns:
        pd.DataFrame: Updated Tensorboard logs
    """
    tb_logs["dir_name"] = tb_logs["dir_name"].map(lambda x: split_dir_name(x))
    return tb_logs


def plot_ap_figures(tb_logs: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    best_ap_data = get_best_ap_data(tb_logs)
    sns.lineplot(data=best_ap_data, x="step", y="AP",
                 hue=best_ap_data["model"]).set_title("AP")
    plt.subplot(1, 2, 2)
    ap50_data = get_ap_data(tb_logs)
    sns.lineplot(data=ap50_data, x="step", y="AP50",
                 hue=ap50_data["model"]).set_title("AP50")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ap_model_comparison.png"))
    plt.close()


def plot_loss_figures(tb_logs: pd.DataFrame, output_dir: str):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(8, 8)
    ax1 = fig.add_subplot(gs[:4, :4])
    ax1.set_ylim(0,0.05)
    ax2 = fig.add_subplot(gs[:4, 4:])
    ax2.set_ylim(0,2)
    ax3 = fig.add_subplot(gs[4:8, 1:7])
    ax3.set_ylim(0,5)

    box_loss_data = get_box_loss_data(tb_logs)
    sns.lineplot(data=box_loss_data, x="step", y="box_loss",
                 hue=box_loss_data["model"], ax=ax1).set_title("box loss")

    cls_loss_data = get_cls_loss_data(tb_logs)
    sns.lineplot(data=cls_loss_data, x="step", y="cls_loss",
                 hue=cls_loss_data["model"], ax=ax2).set_title("class loss")

    total_loss_data = get_total_loss_data(tb_logs)
    sns.lineplot(data=total_loss_data, x="step", y="total_loss",
                 hue=total_loss_data["model"], ax=ax3).set_title("total loss")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_model_comparison.png"))
    plt.close()


if __name__ == "__main__":
    log_dir = "Specify tensorboard directory"
    output_dir = "Specify output directory"
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    logs_df = reader.tensors
    columns = ["best_AP", "AP", "total_loss", "box_loss", "cls_loss"]
    logs_df = utils.filter_validation_logs(logs_df, columns)
    logs_df = dir_name_2_model_name(logs_df)

    plot_ap_figures(logs_df, output_dir)
    plot_loss_figures(logs_df, output_dir)

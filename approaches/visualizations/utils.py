import pandas as pd


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
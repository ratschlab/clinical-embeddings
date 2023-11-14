import gc
import logging

import numpy as np
import pandas as pd
import tables
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from icu_benchmarks.common import constants


def gather_cat_values(common_path, cat_values):
    # not too many, so read all of them
    df_cat = pd.read_parquet(common_path, columns=list(cat_values))

    d = {}
    for c in df_cat.columns:
        d[c] = [x for x in df_cat[c].unique() if not np.isnan(x)]
    return d


def gather_stats_over_dataset(
    parts,
    to_standard_scale,
    to_min_max_scale,
    train_split_pids,
    fill_string,
    clipping_quantile=0,
):
    minmax_scaler = MinMaxScaler()
    bottom_quantile, top_quantile = None, None
    counts = np.zeros(len(to_standard_scale) + len(to_min_max_scale))
    if clipping_quantile > 0:
        assert clipping_quantile < 0.5

        logging.info("Stats: Counting elemets")
        # we first collect counts
        for p in parts:
            df_part = pd.read_parquet(
                p,
                engine="pyarrow",
                columns=to_min_max_scale + to_standard_scale,
                filters=[(constants.PID, "in", train_split_pids)],
            )
            df_part = df_part.replace(np.inf, np.nan).replace(-np.inf, np.nan)
            counts += df_part.notnull().sum().values
            gc.collect()

        quantile_width = (counts * clipping_quantile).astype(int)

        largest_quantile = quantile_width.max()
        top_values = (
            -np.ones((largest_quantile, len(to_standard_scale) + len(to_min_max_scale)))
            * np.inf
        )
        bottom_values = (
            np.ones((largest_quantile, len(to_standard_scale) + len(to_min_max_scale)))
            * np.inf
        )

        logging.info("Stats: Finding quantiles")
        # we gather top-quantile_width values for each columns
        for p in parts:
            df_part = pd.read_parquet(
                p,
                engine="pyarrow",
                columns=to_min_max_scale + to_standard_scale,
                filters=[(constants.PID, "in", train_split_pids)],
            )
            df_part = df_part.replace(np.inf, np.nan).replace(-np.inf, np.nan)
            top_values = np.concatenate(
                [df_part.replace(np.nan, -np.inf).values, top_values], axis=0
            )
            top_values = -np.partition(-top_values, largest_quantile, axis=0)[
                :largest_quantile
            ]
            bottom_values = np.concatenate(
                [df_part.replace(np.nan, np.inf).values, bottom_values], axis=0
            )
            bottom_values = np.partition(bottom_values, largest_quantile, axis=0)[
                :largest_quantile
            ]
            gc.collect()

        top_quantile = -np.sort(-top_values, axis=0)[
            np.clip(quantile_width - 1, 0, np.inf).astype(int),
            np.arange(len(to_standard_scale) + len(to_min_max_scale)),
        ]
        bottom_quantile = np.sort(bottom_values, axis=0)[
            np.clip(quantile_width - 1, 0, np.inf).astype(int),
            np.arange(len(to_standard_scale) + len(to_min_max_scale)),
        ]

        # If no record for the measure we set quantiles to max values -inf, +inf
        top_quantile[np.where(top_quantile == -np.inf)] = np.inf
        bottom_quantile[np.where(bottom_quantile == np.inf)] = -np.inf

    logging.info("Stats: Finding Min-Max")
    for p in parts:
        df_part = impute_df(
            pd.read_parquet(
                p,
                engine="pyarrow",
                columns=to_min_max_scale + [constants.PID],
                filters=[(constants.PID, "in", train_split_pids)],
            ),
            fill_string=fill_string,
        )
        df_part = df_part.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        if clipping_quantile > 0:
            df_part[to_min_max_scale] = df_part[to_min_max_scale].clip(
                bottom_quantile[: len(to_min_max_scale)],
                top_quantile[: len(to_min_max_scale)],
            )
        minmax_scaler.partial_fit(df_part[to_min_max_scale])
        gc.collect()

    means = []
    stds = []
    # cannot read all to_standard_scale columns in memory, one-by-one would very slow, so read a certain number
    # of columns at a time
    if len(to_standard_scale) > 0:
        logging.info("Stats: Finding Mean-Std")
        batch_size = 20
        batches = (
            to_standard_scale[pos : pos + batch_size]
            for pos in range(0, len(to_standard_scale), batch_size)
        )
        pos_to_scale = 0
        for s in batches:
            dfs = impute_df(
                pd.read_parquet(
                    parts[0].parent,
                    engine="pyarrow",
                    columns=[constants.PID] + s,
                    filters=[(constants.PID, "in", train_split_pids)],
                ),
                fill_string=fill_string,
            )
            dfs = dfs.replace(np.inf, np.nan).replace(-np.inf, np.nan)
            if clipping_quantile > 0:
                bot_batch_quantile = bottom_quantile[-len(to_standard_scale) :][
                    pos_to_scale : pos_to_scale + batch_size
                ]
                top_batch_quantile = top_quantile[-len(to_standard_scale) :][
                    pos_to_scale : pos_to_scale + batch_size
                ]
                dfs[s] = dfs[s].clip(bot_batch_quantile, top_batch_quantile)
                pos_to_scale += batch_size

            # don't rely on sklearn StandardScaler as partial_fit does not seem to work correctly
            # if in one iteration all values of a column are nan (i.e. the then mean becomes nan)
            means.extend(dfs[s].mean())
            stds.extend(
                dfs[s].std(ddof=0)
            )  # ddof=0 to be consistent with sklearn StandardScalar
            gc.collect()

    # When there is only one measurement we set std to 1 so we don't divide by 0 and just center value.
    stds = np.array(stds)
    stds[np.where(stds == 0.0)] = 1.0
    stds = list(stds)
    return (means, stds), (bottom_quantile, top_quantile), minmax_scaler


def _normalize_cols(df, output_cols):
    cols_to_drop = [
        c for c in set(df.columns).difference(output_cols) if c != constants.PID
    ]
    if cols_to_drop:
        logging.warning(
            f"Dropping columns {cols_to_drop} as they don't appear in output columns"
        )
    df = df.drop(columns=cols_to_drop)

    cols_to_add = sorted(set(output_cols).difference(df.columns))

    if cols_to_add:
        logging.warning(f"Adding dummy columns {cols_to_add}")
        df[cols_to_add] = 0.0

    col_order = [constants.DATETIME] + sorted(
        [c for c in df.columns if c != constants.DATETIME]
    )
    df = df[col_order]

    cmp_list = list(c for c in df.columns if c != constants.PID)
    assert cmp_list == output_cols

    return df


def to_ml(
    save_path,
    parts,
    labels,
    features,
    endpoint_names,
    df_var_ref,
    fill_string,
    output_cols,
    split_path=None,
    random_seed=42,
    scaler="standard",
    clipping_quantile=0,
):
    df_part = pd.read_parquet(parts[0])
    data_cols = df_part.columns

    common_path = parts[0].parent
    df_pid_and_time = pd.read_parquet(
        common_path, columns=[constants.PID, constants.DATETIME]
    )

    # list of patients for every split
    split_ids = get_splits(df_pid_and_time, split_path, random_seed)

    logging.info("Gathering variable types")
    cat_values, binary_values, to_standard_scale, to_min_max_scale = get_var_types(
        data_cols, df_var_ref, scaler
    )
    to_standard_scale = [c for c in to_standard_scale if c in set(output_cols)]
    to_min_max_scale = [c for c in to_min_max_scale if c in set(output_cols)]

    logging.info("Gathering categorical variables possible values")
    cat_vars_levels = gather_cat_values(common_path, cat_values)

    logging.info("Gathering stats for scaling")
    (means, stds), (bot_quant, top_quant), minmax_scaler = gather_stats_over_dataset(
        parts,
        to_standard_scale,
        to_min_max_scale,
        split_ids["train"],
        fill_string,
        clipping_quantile,
    )

    # for every train, val, test split keep how many records
    # have already been written (needed to compute correct window position)
    output_offsets = {}

    features_available = features
    if not features_available:
        features = [None] * len(parts)

    logging.info("Pre-processing per batch")
    for p, l, f in tqdm(zip(parts, labels, features)):
        df = impute_df(pd.read_parquet(p), fill_string=fill_string)

        # split features between historical feature and prsence features
        df_feat = pd.read_parquet(f) if f else pd.DataFrame(columns=[constants.PID])
        feat_names = df_feat.columns
        history_features_name = [
            name for name in feat_names if name.split("_")[0] != "presence"
        ]
        presence_features_name = [
            name for name in feat_names if name.split("_")[0] == "presence"
        ]
        presence_features_name = [
            constants.PID,
            constants.DATETIME,
        ] + presence_features_name
        presence_available = len(presence_features_name) > 2
        if presence_available:
            df_presence = df_feat[presence_features_name]
        else:
            df_presence = pd.DataFrame(columns=[constants.PID])
        if features_available:  # We extracted some features
            df_feat = df_feat[history_features_name]

        df_label = pd.read_parquet(l)[
            [constants.PID, constants.REL_DATETIME] + list(endpoint_names)
        ]
        df_label = df_label.rename(columns={constants.REL_DATETIME: constants.DATETIME})
        df_label[constants.DATETIME] = df_label[constants.DATETIME] / 60.0

        # align indices between labels df and common df
        df_label = df_label.set_index([constants.PID, constants.DATETIME])
        df_label = df_label.reindex(
            index=zip(df[constants.PID].values, df[constants.DATETIME].values)
        )
        df_label = df_label.reset_index()

        for cat_col in cat_values:
            df[cat_col] = pd.Categorical(df[cat_col], cat_vars_levels[cat_col])

        for bin_col in binary_values:
            bin_vals = [0.0, 1.0]
            if bin_col == "sex":
                bin_vals = ["F", "M"]
            df[bin_col] = pd.Categorical(df[bin_col], bin_vals)

        if cat_values:
            df = pd.get_dummies(df, columns=cat_values)
        if binary_values:
            df = pd.get_dummies(df, columns=binary_values, drop_first=True)

        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        # reorder columns and making sure columns correspond to output_cols
        df = _normalize_cols(df, output_cols)

        split_dfs = {}
        split_labels = {}
        split_features = {}
        split_presence = {}
        for split in split_ids.keys():
            split_dfs[split] = df[df[constants.PID].isin(split_ids[split])]
            split_labels[split] = df_label[
                df_label[constants.PID].isin(split_ids[split])
            ]
            split_features[split] = df_feat[
                df_feat[constants.PID].isin(split_ids[split])
            ]
            split_presence[split] = df_presence[
                df_presence[constants.PID].isin(split_ids[split])
            ]

        # windows computation: careful with offset!
        split_windows = {}
        for split, df in split_dfs.items():
            if df.empty:
                split_windows[split] = np.array([])
                continue
            split_windows[split] = get_windows_split(
                df, offset=output_offsets.get(split, 0)
            )

            assert np.all(
                split_windows[split]
                == get_windows_split(
                    split_labels[split], offset=output_offsets.get(split, 0)
                )
            )
            split_dfs[split] = df.drop(columns=[constants.PID])
            split_labels[split] = split_labels[split].drop(columns=[constants.PID])
            split_features[split] = split_features[split].drop(
                columns=[constants.PID, constants.DATETIME]
            )
            split_presence[split] = split_presence[split].drop(
                columns=[constants.PID, constants.DATETIME]
            )

            output_offsets[split] = output_offsets.get(split, 0) + len(df)

        for split_df in split_dfs.values():
            if split_df.empty:
                continue

            if len(to_standard_scale) > 0:
                if clipping_quantile > 0:
                    split_df[to_standard_scale] = split_df[to_standard_scale].clip(
                        bot_quant[-len(to_standard_scale) :],
                        top_quant[-len(to_standard_scale) :],
                    )
                split_df[to_standard_scale] = (
                    split_df[to_standard_scale].values - means
                ) / stds

            if clipping_quantile > 0:
                split_df[to_min_max_scale] = split_df[to_min_max_scale].clip(
                    bot_quant[: len(to_min_max_scale)],
                    top_quant[: len(to_min_max_scale)],
                )

            split_df[to_min_max_scale] = minmax_scaler.transform(
                split_df[to_min_max_scale]
            )
            split_df.replace(np.inf, np.nan, inplace=True)
            split_df.replace(-np.inf, np.nan, inplace=True)

        split_arrays = {}
        label_arrays = {}
        feature_arrays = {}
        presence_arrays = {}
        for split, df in split_dfs.items():
            array_split = df.values
            array_label = split_labels[split].values

            np.place(array_split, mask=np.isnan(array_split), vals=0.0)

            split_arrays[split] = array_split
            label_arrays[split] = array_label

            if features_available:
                array_features = split_features[split].values
                np.place(array_features, mask=np.isnan(array_features), vals=0.0)
                feature_arrays[split] = array_features

            if presence_available:
                array_presence = split_presence[split].values
                np.place(array_presence, mask=np.isnan(array_presence), vals=0.0)
                presence_arrays[split] = array_presence

            assert len(df.columns) == split_arrays[split].shape[1]

        tasks = list(split_labels["train"].columns)

        output_cols = [c for c in df.columns if c != constants.PID]

        feature_names = list(split_features["train"].columns)
        presence_names = list(split_presence["train"].columns)

        save_to_h5_with_tasks(
            save_path,
            output_cols,
            tasks,
            feature_names,
            presence_names,
            split_arrays,
            label_arrays,
            feature_arrays if features_available else None,
            presence_arrays if presence_available else None,
            split_windows,
        )

        gc.collect()


def get_utf8_cols(col_array):
    return np.array([t.decode("utf-8") for t in col_array[:]])


def merge_multiple_horizon_labels(
    save_path, tables_path, label_radicals, horizons, joint_table_idx=0
):
    horizon_cols = {"train": [], "test": [], "val": []}
    h5_tables = [tables.open_file(data_path, "r").root for data_path in tables_path]
    columns = []
    source_joint_table = h5_tables[joint_table_idx]
    all_labels = get_utf8_cols(source_joint_table["labels"]["tasks"])
    hor_labels = [
        rad + "_" + str(horizons[joint_table_idx]) + "Hours" for rad in label_radicals
    ]
    other_labels_idx = [i for i, k in enumerate(all_labels) if k not in hor_labels]
    other_labels_name = all_labels[other_labels_idx]
    columns += list(other_labels_name)

    for split in horizon_cols.keys():
        horizon_cols[split].append(
            source_joint_table["labels"][split][:, other_labels_idx]
        )
    for table, horizon in zip(h5_tables, horizons):
        all_labels = get_utf8_cols(table["labels"]["tasks"])
        labels_name = []
        labels_idx = []
        for radical in label_radicals:
            label_name = radical + "_" + str(horizon) + "Hours"
            label_idx = np.where(all_labels == label_name)[0][0]
            labels_name.append(label_name)
            labels_idx.append(label_idx)

        for split in horizon_cols.keys():
            horizon_cols[split].append(table["labels"][split][:, labels_idx])

        columns += labels_name
    for split in horizon_cols.keys():
        horizon_cols[split] = np.concatenate(horizon_cols[split], axis=-1)

    col_names = get_utf8_cols(source_joint_table["data"]["columns"])
    task_names = columns
    feature_names = get_utf8_cols(source_joint_table["features"]["name_features"])
    presence_names = get_utf8_cols(
        source_joint_table["presence_features"]["name_features"]
    )

    data_dict = {k: source_joint_table["data"][k][:] for k in ["train", "test", "val"]}
    features_dict = {
        k: source_joint_table["features"][k][:] for k in ["train", "test", "val"]
    }
    presence_dict = {
        k: source_joint_table["presence_features"][k][:]
        for k in ["train", "test", "val"]
    }

    patient_windows_dict = {
        k: source_joint_table["patient_windows"][k][:] for k in ["train", "test", "val"]
    }
    label_dict = horizon_cols
    save_to_h5_with_tasks(
        save_path,
        col_names,
        task_names,
        feature_names,
        presence_names,
        data_dict,
        label_dict,
        features_dict,
        presence_dict,
        patient_windows_dict,
    )


def _write_data_to_hdf(
    data, dataset_name, node, f, first_write, nr_cols, expectedrows=1000000
):
    filters = tables.Filters(complevel=5, complib="blosc:lz4")

    if first_write:
        ea = f.create_earray(
            node,
            dataset_name,
            atom=tables.Atom.from_dtype(data.dtype),
            expectedrows=expectedrows,
            shape=(0, nr_cols),
            filters=filters,
        )
        if len(data) > 0:
            ea.append(data)
    elif len(data) > 0:
        node[dataset_name].append(data)


def save_to_h5_with_tasks(
    save_path,
    col_names,
    task_names,
    feature_names,
    presence_names,
    data_dict,
    label_dict,
    features_dict,
    presence_dict,
    patient_windows_dict,
):
    """
    Save a dataset with the desired format as h5.
    Args:
        save_path: Path to save the dataset to.
        col_names: List of names the variables in the dataset.
        task_names: List of names for the tasks in the dataset.
        feature_names: List of names for the features in the dataset.
        presence_names: List of names for the presence features in the dataset.
        data_dict: Dict with an array for each split of the data
        label_dict: (Optional) Dict with each split and labels array in same order as data_dict.
        features_dict: (Optional) Dict with each split and features array in same order as data_dict.
        presence_dict: (Optional) Dict with each split and presence features array in the same order as data_dict.
        patient_windows_dict: Dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
    Returns:
    """

    # data labels windows

    first_write = not save_path.exists()
    mode = "w" if first_write else "a"

    with tables.open_file(save_path, mode) as f:
        if first_write:
            n_data = f.create_group("/", "data", "Dataset")
            f.create_array(
                n_data, "columns", obj=[str(k).encode("utf-8") for k in col_names]
            )
        else:
            n_data = f.get_node("/data")

        splits = ["train", "val", "test"]
        for split in splits:
            _write_data_to_hdf(
                data_dict[split].astype(float),
                split,
                n_data,
                f,
                first_write,
                data_dict["train"].shape[1],
            )

        if label_dict is not None:
            if first_write:
                labels = f.create_group("/", "labels", "Labels")
                f.create_array(
                    labels, "tasks", obj=[str(k).encode("utf-8") for k in task_names]
                )
            else:
                labels = f.get_node("/labels")

            for split in splits:
                _write_data_to_hdf(
                    label_dict[split].astype(float),
                    split,
                    labels,
                    f,
                    first_write,
                    label_dict["train"].shape[1],
                )

        if features_dict is not None:
            if first_write:
                features = f.create_group("/", "features", "Features")
                f.create_array(
                    features,
                    "name_features",
                    obj=[str(k).encode("utf-8") for k in feature_names],
                )
            else:
                features = f.get_node("/features")

            for split in splits:
                _write_data_to_hdf(
                    features_dict[split].astype(float),
                    split,
                    features,
                    f,
                    first_write,
                    features_dict["train"].shape[1],
                )

        if presence_dict is not None:
            if first_write:
                presence_features = f.create_group(
                    "/", "presence_features", "Presence Features"
                )
                f.create_array(
                    presence_features,
                    "name_features",
                    obj=[str(k).encode("utf-8") for k in presence_names],
                )
            else:
                presence_features = f.get_node("/presence_features")

            for split in splits:
                _write_data_to_hdf(
                    presence_dict[split].astype(float),
                    split,
                    presence_features,
                    f,
                    first_write,
                    presence_dict["train"].shape[1],
                )

        if patient_windows_dict is not None:
            if first_write:
                p_windows = f.create_group("/", "patient_windows", "Windows")
            else:
                p_windows = f.get_node("/patient_windows")

            for split in splits:
                _write_data_to_hdf(
                    patient_windows_dict[split].astype(int),
                    split,
                    p_windows,
                    f,
                    first_write,
                    patient_windows_dict["train"].shape[1],
                )

        if not len(col_names) == data_dict["train"].shape[-1]:
            raise Exception(
                "We saved to data but the number of columns ({}) didn't match the number of features {} ".format(
                    len(col_names), data_dict["train"].shape[-1]
                )
            )


def impute_df(df, fill_string="ffill"):
    if fill_string is not None:
        df = df.groupby(constants.PID).apply(lambda x: x.fillna(method=fill_string))
    return df


def get_var_types(columns, df_var_ref, scaler="standard"):
    cat_ref = list(
        df_var_ref[df_var_ref.variableunit == "Categorical"]["metavariablename"].values
    )
    cat_values = [c for c in cat_ref if c in columns]
    binary_values = list(
        np.unique(
            df_var_ref[df_var_ref["metavariableunit"] == "Binary"]["metavariablename"]
        )
    )
    binary_values += ["sex"]
    to_min_max_scale = [constants.DATETIME, "admissiontime"]
    if scaler == "standard":
        to_standard_scale = [
            k
            for k in np.unique(df_var_ref["metavariablename"].astype(str))
            if k not in cat_values + binary_values
        ] + ["age", "height"]
        to_standard_scale = [c for c in to_standard_scale if c in columns]
    elif scaler == "minmax":
        to_standard_scale = []
        to_min_max_scale += [
            k
            for k in np.unique(df_var_ref["metavariablename"].astype(str))
            if k not in cat_values + binary_values
        ] + ["age", "height"]
        to_min_max_scale = [c for c in to_min_max_scale if c in columns]
    else:
        raise Exception("scaler has to be standard or minmax")

    return cat_values, binary_values, to_standard_scale, to_min_max_scale


def get_splits(df, split_path, random_seed):
    if split_path:
        split_df = pd.read_csv(split_path, sep="\t")
        split_ids = {}
        for split in split_df["split"].unique():
            split_ids[split] = split_df.loc[
                split_df["split"] == split, constants.PID
            ].values
    else:
        split_ids = {}
        train_val_ids, split_ids["test"] = train_test_split(
            np.unique(df[constants.PID]), test_size=0.15, random_state=random_seed
        )
        split_ids["train"], split_ids["val"] = train_test_split(
            train_val_ids, test_size=(0.15 / 0.85), random_state=random_seed
        )
    return split_ids


def get_windows_split(df_split, offset=0):
    pid_array = df_split[constants.PID]
    starts = sorted(np.unique(pid_array, return_index=True)[1])
    stops = np.concatenate([starts[1:], [df_split.shape[0]]])
    ids = pid_array.values[starts]
    return np.stack([np.array(starts) + offset, np.array(stops) + offset, ids], axis=1)

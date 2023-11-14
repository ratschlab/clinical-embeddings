import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_rows = 999


def _get_results_df(path_, test="val", verbose=False):
    """Read metrics file."""
    results_file = f"{test}_metrics.pkl"

    gs = []
    if os.path.isdir(path_):
        seeds = np.sort(os.listdir(path_))
        for seed in [s for s in seeds if s.isdigit()]:
            seed = str(seed)
            path_seed = os.path.join(path_, seed)
            met_dir = os.path.join(path_seed, results_file)
            if os.path.isfile(met_dir):
                try:
                    metrics = pickle.load(open(met_dir, "rb"))

                    for key in [
                        key
                        for key in metrics.keys()
                        if "Curve" not in key and "loss" not in key
                    ]:
                        dict_ = {
                            "metric": key,
                            "model": path_.split("/")[-1],
                            "method": path_.split("/")[-2],
                            "value": metrics[key],
                            "run_number": int(seed),
                            "data_split": results_file.split("_")[0],
                        }
                        gs.append(dict_)

                except (EOFError, pickle.UnpicklingError):
                    continue

    df_gs = pd.DataFrame(gs)

    if len(df_gs) == 0:
        return df_gs

    if verbose:
        print(df_gs)

    return df_gs


def _extract_mean_perf(path_, test, verbose=False):
    df_gs = _get_results_df(path_, test)
    if len(df_gs) == 0:
        return df_gs
    results = pd.concat(
        [
            df_gs.groupby(["model", "metric", "method", "data_split"]).mean()[
                ["value", "run_number"]
            ],
            df_gs.groupby(["model", "metric", "method", "data_split"]).std()["value"]
            / np.sqrt(df_gs.run_number.nunique()),
        ],
        axis=1,
    )
    results.columns = ["mean", "run_number", "stderr"]

    if verbose:
        print(results)
    return results


def _performance_results(list_of_paths, test="val"):
    """Extract performance of list of experiments."""
    final_table = pd.DataFrame()

    for path_ in list_of_paths:

        df_gs = _extract_mean_perf(path_=path_, test=test)
        df_gs = df_gs.reset_index()
        final_table = pd.concat([final_table, df_gs], ignore_index=True)

    return final_table


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _extract_method_parameters(df):
    for el in df["method"].unique():
        for k, v in zip(el.split("_")[1::2], el.split("_")[2::2]):
            if k == "mean" or _is_number(k):
                continue
            if _is_number(v):
                v = float(v)
            df.loc[df["method"] == el, k] = v
        for k, v in zip(el.split("_")[0::2], el.split("_")[1::2]):
            if k == "mean" or _is_number(k):
                continue
            if _is_number(v):
                v = float(v)
            df.loc[df["method"] == el, k] = v

        if "emb" in el.split("_")[0]:
            df.loc[df["method"] == el, "embedding"] = el.split("_")[0]
    return df


def performance_results_embeddings(list_of_paths, test=["val", "test"], embedding=None):

    if list_of_paths == []:
        return pd.DataFrame()
    print(list_of_paths[0].split("/")[:-2])
    all_results = []

    for split in test:
        combined_table = _performance_results(list_of_paths, split)
        results = _extract_method_parameters(combined_table)

        results = results.sort_values("mean")
        results = results.rename(
            columns={"mean": f"mean_{split}", "stderr": f"stderr_{split}"}
        )
        results = results.drop(columns=["data_split"])
        all_results.append(results)
    all_results = pd.merge(*all_results)
    if embedding is not None:
        all_results["embedding"] = embedding

    # only select runs with more than 3 seeds
    unfinished = 100 * np.sum(all_results.run_number < 2222) / len(all_results)
    print("{:.2f} percent of runs have less than 3 seeds completed.".format(unfinished))
    all_results = all_results.loc[all_results.run_number >= 2222]
    return all_results


def _listdir_fullpath(d, add_task=None):
    folders = [os.path.join(d, f) for f in os.listdir(d)]
    if add_task is not None:
        folders = [os.path.join(f, add_task) for f in folders]
        folders_ = []
        for f in folders:
            if os.path.exists(f):
                folders_.append(f)
        folders = folders_
    return folders


def best_random_search_results(
    path, task="Mortality_At24Hours", to_separate="modsplit", plot=True, ylim=[0.5, 0.6]
):

    results_df = []
    for archi in sorted(os.listdir(path)):
        results_df.append(
            performance_results_embeddings(
                _listdir_fullpath(os.path.join(path, archi), add_task=task),
                embedding=archi.replace("emb", ""),
            )
        )
    results_df = pd.concat(results_df)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    all_best_results = []

    if to_separate == "merge":
        # if compare performance of aggregation, exclude runs with no modality splitting.
        results_df = results_df.loc[results_df.modsplit != "hiridnosplit"]
        results_df = results_df.loc[results_df.modsplit != "mimicnosplit"]

    for i, archi in enumerate(results_df.embedding.unique()):
        # Analyse each embedding architecture separately
        embedding_df = results_df.loc[results_df.embedding == archi]
        columns, best_results = [], []

        for col_value in sorted(results_df[to_separate].unique()):
            # Separate results by column

            # Count number of successful runs per column value
            if col_value not in embedding_df[to_separate].unique():
                columns.append("\n N = 0")
            else:
                columns.append(
                    "\n N = " + str(embedding_df[to_separate].value_counts()[col_value])
                )
            axes[0].annotate(
                columns[-1],
                [
                    len(columns) - 1.25,
                    ylim[0] - (ylim[1] - ylim[0]) * (0.125 + 0.05 * i),
                ],
                color=f"C{i}",
                annotation_clip=False,
            )

            if col_value not in embedding_df[to_separate].unique():
                continue

            column_df = embedding_df.loc[
                (embedding_df[to_separate] == col_value) & (embedding_df.metric == "PR")
            ]
            # select best PR validation value for this column value
            best_results.append(
                column_df.loc[column_df.index == column_df.mean_val.idxmax()]
            )
        best_results = pd.concat(best_results)
        all_best_results.append(best_results)

        if plot:
            for ax, mean, std in zip(
                axes, ["mean_val", "mean_test"], ["stderr_val", "stderr_test"]
            ):
                ax.bar(
                    best_results[to_separate].values,
                    best_results[mean].values,
                    yerr=best_results[std].values,
                    label=archi,
                    ecolor=f"C{i}",
                    capsize=10,
                    alpha=0.7,
                )
                ax.set_title(mean.replace("mean_", ""))
                ax.set_ylim(ylim)
                ax.grid(alpha=0.3)

    if plot:
        fig.suptitle(task)
        axes[0].legend()

    all_best_results = pd.concat(all_best_results)

    return all_best_results[
        [
            "model",
            "embedding",
            "metric",
            "modsplit",
            "merge",
            "embdepth",
            "emblatent",
            "l1-reg",
            "mean_val",
            "stderr_val",
            "mean_test",
            "stderr_test",
        ]
    ]

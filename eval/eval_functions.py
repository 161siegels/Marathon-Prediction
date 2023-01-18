import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import r2_score


def model_comparison(
    models: List, sex: str, error_metric: str = "mse", select_top_percent: bool = False
):
    summary = load_model_results(models[0], sex, select_top_percent).add_prefix(
        f"{models[0]}_"
    )
    for m in models[1:]:
        summary = summary.join(
            load_model_results(m, sex, select_top_percent).add_prefix(f"{m}_")
        )
    summary.rename(columns={f"{models[0]}_age": "age"}, inplace=True)
    fig = plt.figure()
    summary = summary.set_index("age")[
        [c for c in summary.columns if error_metric in c]
    ]
    summary.columns = [c.replace(f'_{error_metric}', "") for c in summary.columns]
    ax = summary.plot(kind="bar")
    ax.set_xlabel("Age")
    ax.set_ylabel(error_metric.upper())
    gender_string = "Female" if sex == "F" else "Male"
    ax.set_title(f"{error_metric.upper()} Comparison for {gender_string} Data")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f"../output_files/{error_metric}_comp_{gender_string}.png")
    return ax


def load_model_results(model_str: str, sex: str, select_top_percent: bool):
    file_path_prefix = (
        "output_files/top_percent/" if select_top_percent else "output_files/"
    )
    return pd.read_csv(f"../{file_path_prefix}error_df_{model_str}_{sex}.csv")


def r2_grouped(g, actual, predicted):
    r2 = r2_score(g[actual], g[predicted])
    return pd.Series(dict(r2=r2))


if __name__ == "__main__":
    model_comparison(["ridge", "catboost", "numpyro"], "M", "mae")
    model_comparison(["ridge", "catboost", "numpyro"], "F", "mae")

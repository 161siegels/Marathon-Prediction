import argparse
from modeling.model_data import ModelData
from modeling.model import SimpleRidge, CatBoost, NumpyroModel
import matplotlib.pyplot as plt
from eval.eval_functions import model_comparison
import sys
import numpyro

if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(description="Model Marathon Times!")
    parser.add_argument(
        "--gender",
        type=str,
        default="all",
        help="Gender to Evaluate (default is both)",
        required=False,
    )
    parser.add_argument(
        "--model", type=str, default="ridge", help="Model to use", required=False
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune model or load params (if false)",
        required=False,
    )
    parser.add_argument(
        "--select_top_percent",
        action="store_true",
        help="Only use top 50 percent of runners",
        required=False,
    )
    args = parser.parse_args()

    genders = ["M", "F"] if (args.gender == "all") else [args.gender]
    file_path_prefix = (
        "output_files/top_percent/" if args.select_top_percent else "output_files/"
    )
    for gender in genders:
        gender_name = "Male" if gender == "M" else "Female"
        if args.model.lower() == "ridge":
            data = ModelData(
                features=["age", "marathon", "marathon_year"],
                response="time",
                sex=gender,
                min_races=1,
                ohe_cols=["marathon_year", "marathon"],
                select_top_percent=args.select_top_percent,
            )
            mod = SimpleRidge(data=data, sex=gender, file_path_prefix=file_path_prefix)
        elif args.model.lower() == "catboost":
            data = ModelData(
                features=["age", "marathon", "marathon_year", "global_runner_id"],
                response="time",
                sex=gender,
                min_races=1,
                ohe_cols=[],
                select_top_percent=args.select_top_percent,
            )
            mod = CatBoost(
                data=data,
                sex=gender,
                categorical_cols=["marathon_year", "marathon", "global_runner_id"],
                tune=args.tune,
                file_path_prefix=file_path_prefix,
            )
        elif args.model.lower() == "numpyro":
            data = ModelData(
                features=["age", "marathon", "year", "global_runner_id"],
                response="time",
                sex=gender,
                min_races=1,
                ohe_cols=[],
                select_top_percent=args.select_top_percent,
            )
            mod = NumpyroModel(
                data=data,
                ranefs=["marathon", "year", "global_runner_id"],
                sex=gender,
                file_path_prefix=file_path_prefix,
            )
        else:
            raise NotImplementedError(f"No model implemented with name {args.model}")
        mod.fit()
        mod.test_rmse()
        mod.aging_curve(mod.model_name + f" {gender_name} Aging Curve")
        print("Overall Error Metrics")
        print(mod.error_metrics)

        print("Error Metrics By Age")
        print(mod.error_age_df)
        mod.error_age_df.to_csv(f"{file_path_prefix}error_df_{args.model}_{gender}.csv")

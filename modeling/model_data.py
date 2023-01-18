import pandas as pd
import numpy as np
import os
from patsy import dmatrix, build_design_matrices
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ModelData:
    def __init__(
        self,
        features,
        response,
        ohe_cols=["global_runner_id"],
        sex=None,
        min_races=1,
        test_size=0.1,
        valid_size=0.1,
        select_top_percent=False,
    ):
        self.df = None
        self.train_set = None
        self.valid_set = None
        self.select_top_percent = select_top_percent
        self.response = response
        self.features = features
        self.sex = sex
        self.min_races = min_races
        self.ohe_cols = ohe_cols
        self.test_size = test_size
        self.valid_size = valid_size
        self.scaler = StandardScaler()
        self.load_data()
        self.filter()
        self.feature_eng()
        self.train_valid_split()

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_parquet("data_proc/cleaned_data.parquet")
        if self.select_top_percent:
            age_grps = pd.cut(
                self.df["age"].astype(int), bins=[18, 30, 40, 50, 60, 70, 80, 90]
            )
            idx = (
                self.df.groupby(age_grps)["time"].transform(
                    "rank", method="min", pct=True
                )
                <= 0.50
            )
            self.df = self.df.loc[idx]

    def age_spline(
        self,
        col_name: str = "age",
        min_value: int = 18,
        max_value: int = 85,
        degree: int = 5,
    ) -> pd.DataFrame:
        self.df = self.df.loc[(self.df.age >= min_value) & (self.df.age <= max_value)]
        trans = age_spline(col_name, min_value, max_value, degree)
        if col_name in self.features:
            self.features.remove(col_name)
            self.features = self.features + [
                f"AgeSpline_{i}" for i in range(1, degree + 1)
            ]
        self.df = self.df.merge(trans.reset_index())

    def scale_features(self):
        x_unscaled = df[self.features]
        x_scaled = self.scaler.fit_transform(x_unscaled)
        return x_scaled

    def filter(self):
        if self.sex:
            self.df = self.df.loc[lambda x: x.sex == self.sex]
        self.df = self.df.loc[lambda x: x["year"] >= 2000]
        # self.df = self.df.loc[lambda x: x["marathon"].isin(["NYC", "Chicago", "LA"])]
        if self.min_races > 1:
            mult_races = (
                self.df.groupby("global_runner_id")["global_runner_id"].transform(
                    "count"
                )
                >= self.min_races
            )
            self.df = self.df.loc[mult_races]

    def feature_eng(self):
        self.age_spline()
        self.df["marathon_year"] = (
            self.df["marathon"] + "_" + self.df["year"].astype(int).astype("str")
        )
        self.df["num_races"] = self.df.groupby("global_runner_id")[
            "global_runner_id"
        ].transform("count")
        self.ohe()

    def ohe(self):
        for col in self.ohe_cols:
            ohe = pd.get_dummies(self.df[col], prefix=col)
            self.df = pd.concat([self.df, ohe], axis=1)
            self.features.remove(col)
            self.features = self.features + ohe.columns.to_list()

    def train_valid_split(self):
        self.train_set, self.test_set = train_test_split(
            self.df, test_size=self.test_size, random_state=11
        )
        self.train_set, self.valid_set = train_test_split(
            self.train_set,
            test_size=self.valid_size / (1 - self.test_size),
            random_state=11,
        )


def age_spline(
    col_name: str = "age", min_value: int = 18, max_value: int = 85, degree: int = 5
) -> pd.DataFrame:
    trans = dmatrix(
        "cr(x, df) -1", {"x": np.arange(min_value, max_value + 1), "df": degree}
    )
    trans = build_design_matrices(
        [trans.design_info], {"x": np.arange(min_value, max_value + 1), "df": degree}
    )[0]
    trans = pd.DataFrame(
        np.array(trans),
        index=np.arange(min_value, max_value + 1),
        columns=[f"AgeSpline_{i}" for i in range(1, degree + 1)],
    )
    trans.index.name = col_name
    return trans

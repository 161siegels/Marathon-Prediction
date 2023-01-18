from modeling.model_data import ModelData, age_spline
from modeling import catboost_tuning
from eval.eval_functions import r2_grouped
import abc
from sklearn.linear_model import RidgeCV, LinearRegression
from scipy import sparse
from catboost import CatBoostRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpyro
import numpyro.optim as optim
from sklearn.preprocessing import OrdinalEncoder

from numpyro.infer import MCMC, NUTS, Predictive, initialization, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from jax import random
import jax.numpy as jnp
import json
from typing import List, Dict


class Model:
    def __init__(self, data: ModelData, sex: str, file_path_prefix: str):
        self.data = data
        self.sex = sex
        self.model_name = None
        self.model = None
        self.error_metrics = {}
        self.error_age_df = None
        self.file_path_prefix = file_path_prefix

    @abc.abstractmethod
    def fit(self):
        return

    @abc.abstractmethod
    def predict(self):
        return

    def test_rmse(self):
        self.data.test_set[f"predicted_{self.data.response}"] = self.model.predict(
            self.data.test_set[self.data.features]
        )
        self.error_metrics["mse"] = mean_squared_error(
            self.data.test_set[self.data.response],
            self.data.test_set[f"predicted_{self.data.response}"],
        )
        self.error_metrics["mae"] = mean_absolute_error(
            self.data.test_set[self.data.response],
            self.data.test_set[f"predicted_{self.data.response}"],
        )
        self.error_metrics["r2"] = r2_score(
            self.data.test_set[self.data.response],
            self.data.test_set[f"predicted_{self.data.response}"],
        )

        self.data.test_set["mse"] = (
            self.data.test_set[f"predicted_{self.data.response}"]
            - self.data.test_set[self.data.response]
        ) ** 2
        self.data.test_set["mae"] = np.abs(
            self.data.test_set[f"predicted_{self.data.response}"]
            - self.data.test_set[self.data.response]
        )

        age_grps = pd.cut(
            self.data.test_set["age"].astype(int), bins=[18, 30, 40, 50, 60, 70, 90]
        )
        self.error_age_df = (
            self.data.test_set.groupby(age_grps)[["mse", "mae"]]
            .mean()
            .join(self.data.test_set.groupby(age_grps).size().rename("count"))
            .join(self.data.test_set.groupby(age_grps).apply(r2_grouped, self.data.response, f"predicted_{self.data.response}"))
        )
        self.data.test_set.to_parquet(
            f"{self.file_path_prefix}{self.model_name}_predictions.parquet"
        )

    @abc.abstractmethod
    def aging_curve(self):
        return


class NumpyroModel(Model):
    def __init__(
        self,
        data: ModelData,
        sex: str = None,
        ranefs: List[str] = None,
        gender: str = None,
        file_path_prefix: str = "output_files/",
    ):
        super().__init__(data, sex, file_path_prefix)
        assert set(ranefs) <= set(self.data.features)
        self.model_name = "Numpyro"
        self.gender = gender
        self.ranefs = ranefs
        self.fixefs = [c for c in self.data.features if c not in ranefs]
        self.marathon_encoder = None
        self.runner_encoder = None
        self.output_dict = {}

    def model_fn(
        self,
        intercept: jnp.array,
        n: int,
        n_ages: int,
        age: jnp.array,
        n_races: int,
        n_runners: int,
        n_years: int,
        runner: jnp.array,
        marathon: jnp.array,
        year: jnp.array,
        response: jnp.array,
    ):
        intercept = numpyro.deterministic("intercept", intercept)
        sd = numpyro.sample("sigma", dist.HalfNormal(50.0))
        runner_mu = numpyro.sample("runner_mu_global", dist.Normal(0.0, 1))
        runner_sd = numpyro.sample("runner_sd_global", dist.HalfNormal(100))
        race_mu_global = numpyro.sample("race_mu_global", dist.Normal(0.0, 1))
        race_sd_global = numpyro.sample("race_sd_global", dist.HalfNormal(5))

        n_runners = len(self.data.df["global_runner_id"].unique())

        with numpyro.plate("plate_i", n_runners):
            runner_effect = numpyro.sample("runner", dist.Normal(runner_mu, runner_sd))
        with numpyro.plate("age_dim", n_ages):
            age_sd = numpyro.sample("age_sd", dist.HalfNormal(100))
            age_effect = numpyro.sample("age", dist.Normal(0, age_sd))

        with numpyro.plate("marathon", n_races, dim=-2):
            race_mu = numpyro.sample("mu_race", dist.Normal(0, race_sd_global))
            race_sd = numpyro.sample("sigma_race", dist.HalfNormal(10))
            with numpyro.plate("marathon_year", n_years, dim=-1):
                race_year_mu = numpyro.sample(
                    "mu_race_year", dist.Normal(race_mu_global, race_sd)
                )
                # race_year_mu = numpyro.sample("mu_race_year", dist.Normal(race_mu_global, race_sd_global))

        mean_est = (
            intercept
            + runner_effect[runner]
            + race_mu[marathon].reshape(1, -1)
            + race_year_mu[marathon, year]
            + jnp.sum(age * age_effect, axis=1)
        )
        # mean_est = intercept + runner_effect[runner] + jnp.sum(age*age_effect, axis=1)

        numpyro.sample("obs", dist.TruncatedNormal(mean_est, sd, low=0), obs=response)

    def fit(self):
        self.data.train_set = pd.concat(
            [self.data.train_set, self.data.valid_set], axis=0
        )
        self.marathon_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=len(self.data.train_set["marathon"].unique()),
        )
        self.runner_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=len(self.data.train_set["global_runner_id"].unique()),
        )
        # encode marathons w label encoder
        self.marathon_encoder.fit(self.data.train_set["marathon"].values.reshape(-1, 1))
        self.runner_encoder.fit(
            self.data.train_set["global_runner_id"].values.reshape(-1, 1)
        )
        year_vals = self.data.train_set["year"] - self.data.df["year"].min()
        guide = AutoNormal(self.model_fn)

        # convert to jax
        runners = jnp.array(
            self.runner_encoder.transform(
                self.data.train_set["global_runner_id"].values.reshape(-1, 1)
            ).astype(int)
        ).reshape(1, -1)
        marathon_vals = jnp.array(
            self.marathon_encoder.transform(
                self.data.train_set["marathon"].values.reshape(-1, 1)
            ).astype(int)
        ).reshape(1, -1)
        year_vals = jnp.array(year_vals.astype(int))
        resp = jnp.array(self.data.train_set[self.data.response].values)
        age = jnp.array(self.data.train_set[self.fixefs].values)
        svi = SVI(
            self.model_fn,
            guide,
            optim.Adam(1),
            Trace_ELBO(),
            n=len(self.data.train_set),
            intercept=self.data.train_set["time"].mean(),
            n_ages=len(self.fixefs),
            age=age,
            n_races=self.marathon_encoder.unknown_value,
            n_runners=self.runner_encoder.unknown_value,
            n_years=len(self.data.df["year"].unique()),
            runner=runners,
            marathon=marathon_vals,
            year=year_vals,
            response=resp,
        )
        svi_result = svi.run(
            random.PRNGKey(0), (5_000 if self.data.select_top_percent else 10_000)
        )
        params = svi_result.params
        # posterior_samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
        # dist_posterior = Predictive(model=guide, params=params, num_samples=10000)
        # posterior_samples = dist_posterior(random.PRNGKey(1))

        # pred = np.sum(age*posterior_samples['age'].mean(axis=0), axis=1) + posterior_samples['intercept'].mean() + \
        #        posterior_samples['runner'].mean(axis=0)[runners] + \
        #         posterior_samples['mu_race_year'].mean(axis=0)[marathon_vals, year_vals]
        self.save_model(guide, params, self.marathon_encoder, self.runner_encoder)

    def save_model(self, guide, params, marathon_encoder, runner_encoder):
        self.output_dict["guide"] = guide
        self.output_dict["params"] = params

    def aging_curve(self, plot_name: str):
        age_spline_df = age_spline(min_value=18, max_value=85)
        posterior_samples = self.output_dict["guide"].sample_posterior(
            random.PRNGKey(1), self.output_dict["params"], (1500,)
        )
        preds = (
            np.sum(age_spline_df.values * posterior_samples["age"].mean(axis=0), axis=1)
            + posterior_samples["intercept"].mean()
            + posterior_samples["race_mu_global"].mean(axis=0)
            + posterior_samples["runner_mu_global"].mean(axis=0)
        )
        lmvar = (
            np.matrix(age_spline_df @ np.cov(posterior_samples["age"].T))
            @ age_spline_df.T
        )
        lmsd = np.sqrt(np.diag(lmvar))
        pd.Series(preds, index=np.arange(18, 86), name="marathon_time").to_csv(
            f"{self.file_path_prefix}/marathon_times_{plot_name}.csv"
        )
        plt.plot(range(18, 86), preds)
        # plt.plot(range(18, 86), preds-2*lmsd)
        plt.xlabel("Age")
        plt.ylabel("Marathon Time (minutes)")
        plt.title(plot_name)
        plt.savefig(f"{self.file_path_prefix}{plot_name}.png")
        plt.clf()

    def save_coefficients(self, posterior_samples: Dict, year_vals: np.array):
        marathon_df = pd.DataFrame(
            posterior_samples["mu_race"].mean(axis=0),
            index=self.marathon_encoder.categories_,
        )
        marathon_year_df = pd.DataFrame(
            posterior_samples["mu_race_year"].mean(axis=0),
            index=self.marathon_encoder.categories_,
            columns=(np.arange(0, max(year_vals) + 1) + self.data.df["year"].min()),
        )
        marathon_df.to_csv(
            f"{self.file_path_prefix}{self.model_name}_{self.sex}_marathon_coef.csv"
        )
        marathon_year_df.to_csv(
            f"{self.file_path_prefix}{self.model_name}_{self.sex}_marathon_year_coef.csv"
        )

    def test_rmse(self):
        # HANDLE UNKNOWN
        year_vals = self.data.test_set["year"] - self.data.df["year"].min()
        runners = jnp.array(
            self.runner_encoder.transform(
                self.data.test_set["global_runner_id"].values.reshape(-1, 1)
            ).astype(int)
        ).reshape(1, -1)
        marathon_vals = jnp.array(
            self.marathon_encoder.transform(
                self.data.test_set["marathon"].values.reshape(-1, 1)
            ).astype(int)
        ).reshape(1, -1)
        year_vals = jnp.array(year_vals.astype(int))
        age = jnp.array(self.data.test_set[self.fixefs].values)
        posterior_samples = self.output_dict["guide"].sample_posterior(
            random.PRNGKey(1), self.output_dict["params"], (1500,)
        )
        # add default runner
        runner_coefs = np.append(
            posterior_samples["runner"].mean(axis=0),
            posterior_samples["runner_mu_global"].mean(),
        )
        # runner_coefs = np.append(posterior_samples['runner'].mean(axis=0), 0)
        self.data.test_set[f"predicted_{self.data.response}"] = (
            np.sum(age * posterior_samples["age"].mean(axis=0), axis=1)
            + posterior_samples["intercept"].mean()
            + runner_coefs[runners][0]
            + posterior_samples["mu_race_year"].mean(axis=0)[marathon_vals, year_vals][
                0
            ]
            + posterior_samples["mu_race"].mean(axis=0)[marathon_vals][0].squeeze()
        )
        self.error_metrics["mse"] = mean_squared_error(
            self.data.test_set[self.data.response],
            self.data.test_set[f"predicted_{self.data.response}"],
        )
        self.error_metrics["mae"] = mean_absolute_error(
            self.data.test_set[self.data.response],
            self.data.test_set[f"predicted_{self.data.response}"],
        )
        self.error_metrics["r2"] = r2_score(
            self.data.test_set[self.data.response],
            self.data.test_set[f"predicted_{self.data.response}"],
        )
        self.data.test_set["mse"] = (
            self.data.test_set[f"predicted_{self.data.response}"]
            - self.data.test_set[self.data.response]
        ) ** 2
        self.data.test_set["mae"] = np.abs(
            self.data.test_set[f"predicted_{self.data.response}"]
            - self.data.test_set[self.data.response]
        )
        age_grps = pd.cut(
            self.data.test_set["age"].astype(int), bins=[18, 30, 40, 50, 60, 70, 90]
        )
        self.error_age_df = (
            self.data.test_set.groupby(age_grps)[["mse", "mae"]]
            .mean()
            .join(self.data.test_set.groupby(age_grps).size().rename("count"))
            .join(self.data.test_set.groupby(age_grps).apply(r2_grouped, self.data.response, f"predicted_{self.data.response}"))
        )
        self.data.test_set.to_parquet(
            f"{self.file_path_prefix}{self.model_name}_{self.sex}_predictions.parquet"
        )
        self.save_coefficients(posterior_samples, year_vals)

    def coef(self):
        posterior_samples = self.output_dict["guide"].sample_posterior(
            random.PRNGKey(1), self.output_dict["params"], (1500,)
        )
        runners = pd.Series(
            posterior_samples["runner"].mean(axis=0), self.runner_encoder.classes_
        )


class SimpleRidge(Model):
    def __init__(
        self, data: ModelData, sex: str = None, file_path_prefix: str = "output_files/"
    ):
        super().__init__(data, sex, file_path_prefix)
        self.model_name = "Ridge"
        self.model = RidgeCV()

    def fit(self):
        self.data.train_set = pd.concat(
            [self.data.train_set, self.data.valid_set], axis=0
        )
        self.model.fit(
            self.data.train_set[self.data.features],
            self.data.train_set[self.data.response],
        )
        self.coefficients = pd.Series(self.model.coef_, index=self.data.features)

    def aging_curve(self, plot_name: str):
        age_spline_df = age_spline(min_value=18, max_value=85)
        age_spline_df = age_spline_df.assign(
            **{c: 0 for c in self.coefficients.index if c not in age_spline_df.columns}
        )
        preds = self.model.predict(age_spline_df)
        pd.Series(preds, index=np.arange(18, 86), name="marathon_time").to_csv(
            f"{self.file_path_prefix}/marathon_times_{plot_name}.csv"
        )
        plt.plot(range(18, 86), preds)
        plt.xlabel("Age")
        plt.ylabel("Marathon Time (minutes)")
        plt.title(plot_name)
        plt.savefig(f"{self.file_path_prefix}{plot_name}.png")
        plt.clf()


class CatBoost(Model):
    def __init__(
        self,
        data: ModelData,
        categorical_cols: List[str],
        sex: str = None,
        tune: bool = False,
        file_path_prefix="output_files/",
    ):
        super().__init__(data, sex, file_path_prefix)
        self.model = CatBoostRegressor()
        self.model_name = "CatBoost"
        self.categorical_cols = categorical_cols
        self.tune = tune

    def fit(self):
        if self.tune:
            best_params = catboost_tuning.tune(
                self.data.train_set[self.data.features],
                self.data.train_set[self.data.response],
                self.data.valid_set[self.data.features],
                self.data.valid_set[self.data.response],
                cat_features=self.categorical_cols,
            )
            with open(f"{self.file_path_prefix}params_{self.sex}.json", "w") as fp:
                json.dump(best_params, fp)
        else:
            with open(f"{self.file_path_prefix}params_{self.sex}.json") as json_file:
                best_params = json.load(json_file)
        self.model = CatBoostRegressor(**best_params)
        # self.data.train_set = pd.concat([self.data.train_set, self.data.valid_set], axis=0)
        self.model.fit(
            self.data.train_set[self.data.features],
            self.data.train_set[self.data.response],
            cat_features=self.categorical_cols,
        )

    def aging_curve(self, plot_name: str):
        age_spline_df = age_spline(min_value=18, max_value=85)
        age_spline_df = age_spline_df.assign(
            **{c: "-1" for c in self.data.features if c not in age_spline_df.columns}
        )
        age_spline_df[self.categorical_cols] = age_spline_df[
            self.categorical_cols
        ].astype("category")
        preds = self.model.predict(age_spline_df[self.model.feature_names_])
        pd.Series(preds, index=np.arange(18, 86), name="marathon_time").to_csv(
            f"{self.file_path_prefix}/marathon_times_{plot_name}.csv"
        )
        plt.plot(range(18, 86), preds)
        plt.xlabel("Age")
        plt.ylabel("Marathon Time (minutes)")
        plt.legend()
        plt.title(plot_name)
        plt.savefig(f"{self.file_path_prefix}{plot_name}.png")
        plt.clf()
        # for i in self.data.df.marathon.unique():
        #     age_spline_df = age_spline(min_value=18, max_value=90)
        #     age_spline_df = age_spline_df.assign(**{c:i for c in self.data.features if c not in age_spline_df.columns})
        #     age_spline_df[self.categorical_cols] = age_spline_df[self.categorical_cols].astype('category')
        #     preds = self.model.predict(age_spline_df[self.model.feature_names_])
        #     plt.plot(range(18, 90), preds, label = i)
        #     plt.xlabel("Age")
        #     plt.ylabel("Marathon Time (minutes)")
        # plt.legend()
        # plt.show()

        # marathon = self.data.df.marathon.unique()[0]
        # for i in self.data.df[lambda x: x['marathon'] == marathon].marathon_year.unique():
        #     age_spline_df = age_spline(min_value=18, max_value=90)
        #     age_spline_df = age_spline_df.assign(**{c:i for c in self.data.features if c not in age_spline_df.columns})
        #     age_spline_df[self.categorical_cols] = age_spline_df[self.categorical_cols].astype('category')
        #     preds = self.model.predict(age_spline_df[self.model.feature_names_])
        #     plt.plot(range(18, 90), preds, label = i)
        #     plt.xlabel("Age")
        #     plt.ylabel("Marathon Time (minutes)")
        # plt.legend()
        # plt.show()

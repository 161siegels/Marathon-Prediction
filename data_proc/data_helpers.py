import pandas as pd


def get_runners(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = reidentify_runners(df)
    df = fix_bad_entries(df)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Country"] = df["Country"].fillna("USA")
    # clean name a little
    df = name_cleaning(df)

    df["Age"] = df["Age"].astype(float)
    # for now, dropping data if no age. Later maybe can fill in with age group
    df.dropna(subset=["Age"], inplace=True)
    # Remove races that are > 6.5 hours. Running standard
    df = df.loc[(df["Time"] <= 60 * 6.5)]
    # Remove ages below 19 and above 90
    df = df.loc[(df["Age"] <= 90) & (df["Age"] >= 19)]
    # drop meaningless cols
    df.drop(
        columns=["AgeText", "Bar", "index"],
    )
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def name_cleaning(df: pd.DataFrame) -> pd.DataFrame():
    df["Name"] = (
        df["Name"]
        .str.capitalize()
        .str.replace('"', "")
        .str.replace("  ", " ")
        .str.replace("Ø", "O")
        .str.replace(".", "")
        .str.replace("'", "")
    )
    df["Name"] = df["Name"].str.strip()
    df["Name"] = (
        df["Name"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    # remove no-names
    df = df[df["Name"] != "#value! #value!"]
    return df


def reidentify_runners(df: pd.DataFrame) -> pd.DataFrame:
    # get age of runner if each race was in 2000
    # Useful for joining
    df["age_2000"] = 2000 - df["year"] + df["age"]
    # first let's remove duplicates from same year and race
    # bc impossible to distinguish
    duplicates = df.duplicated(
        subset=["name", "age", "sex", "marathon", "year"], keep=False
    )
    print(f"Same race duplicates: {duplicates.mean()}")
    df = df.loc[~duplicates]

    # Here I will remove any occurences of repeat names in same year
    # where age was either 1 year older or younger
    # unfortunate to remove but makes the re-identification much safer
    for diff in [1, -1]:
        df[f"age_diff_{diff}"] = df["age"] + diff
        df_age_diff = df.copy()
        df_age_diff["age"] = df_age_diff["age"] + diff
        df = pd.concat([df, df_age_diff], axis=0)
        duplicates_diff = df.duplicated(
            subset=["name", "age", "sex", "marathon", "year"], keep=False
        )
        print(f"Same race duplicates with age diff of {diff}: {duplicates_diff.mean()}")
        df = df.loc[~duplicates_diff]
        df = df.loc[df["age"] != df[f"age_diff_{diff}"]]

    # create a runner_marathon_id
    df["runner_marathon_id"] = df.groupby(
        ["name", "marathon", "sex", "age_2000"]
    ).ngroup()

    # Runner could either be 1 year old or younger in same year depending
    # on race timing. Account for that and join
    df[f"age_diff_1"] = df["age_2000"] + 1
    df[f"age_diff_-1"] = df["age_2000"] - 1
    df_age_diff_1 = df.copy()
    df_age_diff_neg1 = df.copy()
    df_age_diff_1["age_2000"] = df_age_diff_1["age_2000"] + 1
    df_age_diff_neg1["age_2000"] = df_age_diff_neg1["age_2000"] - 1
    df = pd.concat([df, df_age_diff_1, df_age_diff_neg1], axis=0)
    df["global_runner_id"] = df.groupby(["name", "sex", "age_2000"]).ngroup()

    # keep the globalrunnerid that has most matches for each marathonrunnerid
    df["global_runner_id_count"] = df.groupby("global_runner_id")[
        "global_runner_id"
    ].transform("count")
    df = df.sort_values("global_runner_id").reset_index(drop=True)
    global_runner_ids = df["global_runner_id"].loc[
        df.groupby(["runner_marathon_id"])["global_runner_id_count"].idxmax()
    ]
    df = df.loc[df["global_runner_id"].isin(global_runner_ids)]
    df = df.groupby(["runner_marathon_id", "year"]).first().reset_index()

    # Ensure no runners have 3 ages in same year
    errors = (
        df.groupby(["global_runner_id", "year"])["age"].transform("max")
        - df.groupby(["global_runner_id", "year"])["age"].transform("min")
    ) > 1
    errors = errors | (
        (
            df.groupby(["global_runner_id", "year"])["time"].transform("max")
            - df.groupby(["global_runner_id", "year"])["time"].transform("min")
        )
        > 100
    )
    print(f"Same year duplicates with age diff greater than 1: {errors.mean()}")
    df = df[~errors]
    global_runner_id_count = df.groupby("global_runner_id")["global_runner_id"].count()
    print(
        f"Number of unique runners that ran more than once: {(global_runner_id_count>1).sum()}"
    )
    print(
        f"Percentage of runners who run multiple races: {(global_runner_id_count>1).sum()/len(global_runner_id_count)}"
    )
    df.drop(columns=["age_diff_1", "age_diff_-1"], inplace=True)
    return df


def fix_bad_entries(df: pd.DataFrame):
    bad_entries = find_bad_rows(df, "M", 85, 250)
    bad_entries = bad_entries | find_bad_rows(df, "M", 80, 210)
    bad_entries = bad_entries | find_bad_rows(df, "M", 75, 200)
    bad_entries = bad_entries | find_bad_rows(df, "M", 70, 190)
    bad_entries = bad_entries | find_bad_rows(df, "M", 65, 170)
    bad_entries = bad_entries | find_bad_rows(df, "M", 60, 160)
    bad_entries = bad_entries | find_bad_rows(df, "F", 85, 320)
    bad_entries = bad_entries | find_bad_rows(df, "F", 80, 260)
    bad_entries = bad_entries | find_bad_rows(df, "F", 75, 230)
    bad_entries = bad_entries | find_bad_rows(df, "F", 70, 215)
    bad_entries = bad_entries | find_bad_rows(df, "F", 65, 200)
    bad_entries = bad_entries | find_bad_rows(df, "F", 60, 185)
    print("Number of bad entries: ", bad_entries.sum())
    df = df.loc[~bad_entries]
    df.loc[(df.name == "Veerabhadra gundu") & (df.age > 80), "age"] += -40
    return df.loc[~bad_entries]


def find_bad_rows(df: pd.DataFrame, sex: str, min_age: float, time: float):
    return (df.sex == sex) & (df.age >= min_age) & (df.time <= time)

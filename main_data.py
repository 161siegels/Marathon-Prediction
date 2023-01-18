from data_proc.data_helpers import get_runners
import pandas as pd

if __name__ == "__main__":
    # marathon_data = pd.read_parquet(f"data_proc/Marathon_Data.parquet")
    marathon_data = pd.read_parquet(f"data_proc/all_marathon_data.parquet")
    df = get_runners(df=marathon_data)
    df.to_parquet("data_proc/cleaned_data.parquet")

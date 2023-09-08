import pandas as pd


## from crisis_helper.params import * ==> still need to do the params.py file

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - removing duplicates
    """

    # Remove duplicates rows based on tweet_text column
    cleaned_df = df.drop_duplicates(subset=['tweet_text']).reset_index(drop=True)

    print("✅ data cleaned")

    return cleaned_df

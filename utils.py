
import pandas as pd
def get_string_cols(df):
    return df.select_dtypes(include=['object']).columns.tolist()

def factorize(df, exclude = []):
    for col in get_string_cols(df):

        if col not in exclude:

            df[col] = pd.factorize(df[col])[0]
    return df

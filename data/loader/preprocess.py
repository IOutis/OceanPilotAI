import pandas as pd


def null_imputation(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Impute or drop null values based on threshold.
    
    Args:
        df: Input DataFrame
        threshold: Proportion of allowed nulls in a column (default 0.5)

    Returns:
        DataFrame with nulls handled
    """
    # Drop columns with too many nulls
    drop_cols = [col for col in df.columns if df[col].isnull().mean() > threshold]
    df.drop(columns=drop_cols, inplace=True)

    # Handle remaining nulls
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna('Unknown', inplace=True)
        elif pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].mean(), inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column].fillna(df[column].mode()[0], inplace=True)

    return df

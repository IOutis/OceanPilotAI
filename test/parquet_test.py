import pandas as pd
import geopandas as gpd

def load_parquet_smart(file_path: str):
    """
    Load a Parquet file. If it contains geometry, return GeoDataFrame.
    Otherwise, return DataFrame.
    """
    try:
        # First try GeoPandas
        df = gpd.read_parquet(file_path)
        if "geometry" in df.columns:
            print(f"GeoParquet detected with geometry column.")
            return df
        else:
            print(f"Parquet loaded but no geometry column found.")
            return pd.read_parquet(file_path)
    except Exception as e:
        print(f"GeoPandas failed ({e}), falling back to Pandas.")
        return pd.read_parquet(file_path)


# Example usage
if __name__ == "__main__":
    path = "00a7f490-7418-413b-a9f1-666f186769b6.parquet"
    df = load_parquet_smart(path)
    print(df.head())
    print(type(df))  # will show GeoDataFrame or DataFrame

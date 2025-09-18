from fastapi import UploadFile
import gzip, shutil
import pandas as pd
import io # Import the io module
from wodpy import wod
import io
import wodpy
import tempfile
import os
from pandas import json_normalize

def flatten_record(record, parent_key='', sep='.'):
    """
    Recursively flattens a single dict record so all values are primitive types.
    Nested dicts/lists are expanded with dot notation keys.
    """
    items = {}
    if isinstance(record, dict):
        for k, v in record.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_record(v, new_key, sep=sep))
            elif isinstance(v, list):
                # Flatten lists by joining values as comma-separated string
                if all(isinstance(i, (str, int, float, bool, type(None))) for i in v):
                    items[new_key] = ','.join(str(i) for i in v)
                else:
                    # If list contains dicts, flatten each and join as string
                    items[new_key] = ';'.join(
                        str(flatten_record(i, '', sep=sep)) if isinstance(i, dict) else str(i)
                        for i in v
                    )
            else:
                items[new_key] = v
    else:
        items[parent_key] = record
    return items

def flatten_data(data):
    """
    Flattens a list of dict records so all values are primitive types.
    Returns a new list of flattened dicts.
    """
    return [flatten_record(record) for record in data]


def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens nested structures in a pandas DataFrame.
    - Dicts become separate columns.
    - Lists become exploded into multiple rows.
    """
    # Step 1: Expand dict-like columns
    dict_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, dict)).any()]
    for col in dict_cols:
        expanded = json_normalize(df[col]).add_prefix(f"{col}.")
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    # Step 2: Explode list-like columns
    list_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]
    for col in list_cols:
        df = df.explode(col).reset_index(drop=True)

    return df


async def parse_excel_file(file_content: bytes) -> dict:
    """Parses an Excel file content and returns a data preview."""
    preview = None
    data = None
    try:
        buffer = io.BytesIO(file_content)
        # Attempt to read as Excel file
        df = pd.read_excel(buffer, engine='openpyxl')
        data = df.to_dict(orient='records')
        preview = df.head().to_dict(orient='records')
        print(f"Excel parsed successfully with {len(data)} rows.")
        print(f"Prview Excel parsed successfully with {len(preview)} rows.")
    except Exception as e:
        preview = {"error": f"Could not parse as Excel: {str(e)}"}
    return data,preview

async def parse_json_file(file_content: bytes) -> tuple:
    """Parses a JSON file and returns preview + full data."""
    try:
        buffer = io.BytesIO(file_content)
        df = pd.read_json(buffer, lines=False)
        data = df.to_dict(orient="records")
        preview = df.head().to_dict(orient="records")
        print(f"JSON parsed successfully with {len(data)} rows.")
        return data, preview
    except Exception as e:
        return None, {"error": f"Could not parse as JSON: {str(e)}"}

async def parse_feather_file(file_content: bytes) -> tuple:
    """Parses a Feather file into preview + full data."""
    try:
        buffer = io.BytesIO(file_content)
        df = pd.read_feather(buffer)
        data = df.to_dict(orient="records")
        preview = df.head().to_dict(orient="records")
        print(f"Feather parsed successfully with {len(data)} rows.")
        return data, preview
    except Exception as e:
        return None, {"error": f"Could not parse as Feather: {str(e)}"}

async def parse_hdf5_file(file_content: bytes) -> tuple:
    """Parses an HDF5 file into preview + full data (first key only)."""
    try:
        buffer = io.BytesIO(file_content)
        with pd.HDFStore(buffer) as store:
            keys = store.keys()
            if not keys:
                return None, {"error": "Empty HDF5 file"}
            df = store[keys[0]]
        data = df.to_dict(orient="records")
        preview = df.head().to_dict(orient="records")
        print(f"HDF5 parsed successfully with {len(data)} rows.")
        return data, preview
    except Exception as e:
        return None, {"error": f"Could not parse as HDF5: {str(e)}"}

import xarray as xr

async def parse_netcdf_file(file_content: bytes) -> tuple:
    """Parses a NetCDF file and returns variable summary + sample."""
    try:
        buffer = io.BytesIO(file_content)
        ds = xr.open_dataset(buffer)
        preview = {var: ds[var].values[:3].tolist() for var in list(ds.data_vars)[:5]}
        data = {var: ds[var].values.tolist() for var in list(ds.data_vars)[:5]}  # limit for safety
        print(f"NetCDF parsed with variables: {list(ds.data_vars)}")
        return data, preview
    except Exception as e:
        return None, {"error": f"Could not parse as NetCDF: {str(e)}"}


import pandas as pd
import geopandas as gpd
import io
from fastapi.encoders import jsonable_encoder

import io
import pandas as pd

import io
import pandas as pd
import math

def make_json_safe(obj):
    """Recursively convert objects to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (int, str)) or obj is None:
        return obj
    elif isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        else:
            return None  # convert NaN/inf to None
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode(errors="ignore")
    else:
        return str(obj)

async def parse_parquet_file(file_content: bytes) -> tuple:
    """Parses a Parquet file content and returns JSON-safe preview + data."""
    try:
        buffer = io.BytesIO(file_content)
        df = pd.read_parquet(buffer)

        # Convert each column to JSON-safe types
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, bytes) else x)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(lambda x: float(x) if pd.notnull(x) and math.isfinite(x) else None)
            elif isinstance(df[col].dtype, pd.arrays.NumpyExtensionDtype):
                df[col] = df[col].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)

        # Convert to dict and make JSON-safe recursively
        data = make_json_safe(df.to_dict(orient="records"))
        preview = make_json_safe(df.head().to_dict(orient="records"))

        print(f"Parquet parsed successfully with {len(data)} rows.")
        return data, preview

    except Exception as e:
        return None, {"error": f"Could not parse as Parquet: {str(e)}"}


async def parse_csv_file(file_content: bytes) -> dict:
    """Parses a CSV file content and returns a data preview."""
    preview = None
    data = None
    try:
        buffer = io.BytesIO(file_content)
        # Attempt to decompress if it's gzipped, otherwise read as plain CSV
        try:
            df = pd.read_csv(buffer, compression='gzip')
        except:
            buffer.seek(0) # Reset buffer if gzip fails
            df = pd.read_csv(buffer)
        
        data = df.to_dict(orient='records')
        preview = df.head().to_dict(orient='records')
        print(f"CSV parsed successfully with {len(data)} rows.")
        print(f"Prview CSV parsed successfully with {len(preview)} rows.")
    except Exception as e:
        preview = {"error": f"Could not parse as CSV: {str(e)}"}
    return data,preview

def _process_single_wod_profile(profile: wod.WodProfile) -> pd.DataFrame:
    """
    Takes a single wodpy profile object and transforms it into a "long",
    tabular DataFrame, handling different WOD data structures. This process
    "flattens" the data from lists into individual rows.
    """
    # --- METHOD 1: Try standard helper functions first for all common variables ---
    data_vectors = {
        'depth': profile.z(),
        'temperature': profile.t(),
        'salinity': profile.s(),
        'oxygen': profile.oxygen(),
        'phosphate': profile.phosphate(),
        'silicate': profile.silicate(),
        # 'nitrate': profile.nitrate(),
        'pH': profile.pH(),
        'pressure': profile.p(),
    }
    valid_vectors = {k: v.tolist() for k, v in data_vectors.items() if v is not None}
    
    # --- METHOD 2: If standard methods fail, parse the raw profile_data ---
    if not valid_vectors and profile.profile_data:
        print("Standard accessors failed, parsing raw profile_data.")
        try:
            depths = [level.get('Depth') for level in profile.profile_data]
            # Assuming the primary measurement is the first variable
            values = [level['variables'][0].get('Value') for level in profile.profile_data]
            # Try to get the variable name from metadata
            var_name = profile.var_metadata(1).get('name', 'primary_value').lower()
            
            valid_vectors = {'depth': depths, var_name: values}
        except (IndexError, KeyError, TypeError) as e:
            print(f"Could not parse raw profile_data for a profile: {e}")
            return pd.DataFrame()

    if not valid_vectors:
        return pd.DataFrame()
        
    df = pd.DataFrame(valid_vectors)
    df['latitude'] = float(profile.latitude())
    df['longitude'] = float(profile.longitude())
    df['date'] = str(profile.datetime())
    df['cruise'] = profile.cruise()
    
    return df


# async def parse_wod_file(file_content: bytes) -> dict:
#     """
#     Decompresses a WOD .gz file to ASCII, parses profiles with wodpy,
#     and returns a small preview of metadata + measurements.
#     """
#     temp_ascii_path = "temp_wod_ascii.txt"
#     try:
#         # 1. Write decompressed content to an ASCII file
#         with gzip.open(io.BytesIO(file_content), "rb") as f_in:
#             with open(temp_ascii_path, "wb") as f_out:
#                 shutil.copyfileobj(f_in, f_out)

#         # 2. Open ASCII file for parsing
#         profiles_preview = []
#         profiles = []
#         with open(temp_ascii_path, "r") as fid:
#             while True:
#                 try:
#                     profile = wod.WodProfile(fid)
#                     print("Profile : ",profile.profile_data)
#                     profile_df = _process_single_wod_profile(profile)
#                     if not profile_df.empty:
#                         profiles.append(profile_df)
#                     # variables = []
#                     # for i in range(len(profile.profile_data)):
#                     #     variables.append(profile.profile_data[i]["variables"])
#                     # oxygen =profile.oxygen()
#                     # pH = profile.pH()
#                     # p = profile.p()
#                     # z = profile.z()
#                     # t = profile.t()
#                     # s = profile.s()
#                     # silicate = profile.silicate()
#                     # phosphate = profile.phosphate()
                    
#                     # profiles.append({
#                     #     "cruise": profile.cruise(),
#                     #     "latitude": float(profile.latitude()),
#                     #     "longitude": float(profile.longitude()),
#                     #     "date": str(profile.datetime()),
#                     #     "levels": int(profile.n_levels()),
#                     #     "depths": z.tolist() if z is not None else [],
#                     #     "biological_header": profile.biological_header,  # if this is a dict, it's safe
#                     #     "oxygen": oxygen.tolist() if oxygen is not None else [],
#                     #     "pH": pH.tolist() if pH is not None else [],
#                     #     "pressure": p.tolist() if p is not None else [],
#                     #     # "nitrate": variables[:3],  # double-check if variables contain arrays too
#                     #     "temperatures":t.tolist() if t is not None else [],
#                     #     "salinity":s.tolist() if s is not None else [],
#                     #     "silicate":silicate.tolist() if silicate is not None else [],
#                     #     "phosphate":phosphate.tolist() if phosphate is not None else [],
#                     #     "var":profile.var_metadata(1)
#                     # })
#                     # if len(profiles_preview) < 3:
#                     #     profiles_preview.append({
#                     #         "cruise": profile.cruise(),
#                     #     "latitude": float(profile.latitude()),
#                     #     "longitude": float(profile.longitude()),
#                     #     "date": str(profile.datetime()),
#                     #     "levels": int(profile.n_levels()),
#                     #     "depths": z.tolist()[:3] if z is not None else [],
#                     #     "biological_header": profile.biological_header,  # if this is a dict, it's safe
#                     #     "oxygen": oxygen.tolist()[:3] if oxygen is not None else [],
#                     #     "pH": pH.tolist()[:3] if pH is not None else [],
#                     #     "pressure": p.tolist()[:3] if p is not None else [],
#                     #     # "nitrate": variables[:3],  # double-check if variables contain arrays too
#                     #     "temperatures":t.tolist()[:3] if t is not None else [],
#                     #     "salinity":s.tolist()[:3] if s is not None else [],
#                     #     "silicate":silicate.tolist()[:3] if silicate is not None else [],
#                     #     "phosphate":phosphate.tolist()[:3] if phosphate is not None else [],
#                     #     "var":profile.var_metadata(1)
#                     #     })
#                 except Exception as e:
#                     print(f"Finished parsing profiles or encountered error: {e}")
#                     break
#                     # return profiles, profiles_preview
        
#         final_df = pd.concat(profiles, ignore_index=True)
#         final_df_safe = make_json_safe(final_df)
#         data = final_df_safe.to_dict(orient='records')
#         profiles_preview = final_df_safe.head(3).to_dict(orient='records')
#         print(len(data), len(profiles_preview), type(data), type(profiles_preview))
#         return data, profiles_preview

#     except Exception as e:
#         print(f"Exception during WOD parsing: {e}")
#         return {"error": f"Exception occurred: {e}"}

#     finally:
#         # 3. Clean up temp file
#         if os.path.exists(temp_ascii_path):
#             os.remove(temp_ascii_path)

async def parse_wod_file(file_content: bytes) -> dict:
    """
    Decompresses a WOD .gz file to ASCII, parses profiles with wodpy,
    and returns a small preview of metadata + measurements.
    """
    temp_ascii_path = "temp_wod_ascii.txt"
    try:
        # 1. Write decompressed content to an ASCII file
        with gzip.open(io.BytesIO(file_content), "rb") as f_in:
            with open(temp_ascii_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 2. Open ASCII file for parsing
        profiles = []
        with open(temp_ascii_path, "r") as fid:
            while True:
                try:
                    profile = wod.WodProfile(fid)
                    profile_df = _process_single_wod_profile(profile)
                    if not profile_df.empty:
                        profiles.append(profile_df)
                except Exception as e:
                    print(f"Finished parsing profiles or encountered error: {e}")
                    break

        final_df = pd.concat(profiles, ignore_index=True)
        final_df_safe = final_df.where(pd.notnull(final_df), None)
        data = final_df_safe.to_dict(orient='records')
        profiles_preview = final_df_safe.head(3).to_dict(orient='records')

        # --- FLATTEN and MAKE JSON SAFE ---
        data_flat = [make_json_safe(flatten_record(rec)) for rec in data]
        preview_flat = [make_json_safe(flatten_record(rec)) for rec in profiles_preview]

        print(len(data_flat), len(preview_flat), type(data_flat), type(preview_flat))
        return data_flat, preview_flat

    except Exception as e:
        print(f"Exception during WOD parsing: {e}")
        return {"error": f"Exception occurred: {e}"}

    finally:
        # 3. Clean up temp file
        if os.path.exists(temp_ascii_path):
            os.remove(temp_ascii_path)

async def extract_file_metadata(file: UploadFile) -> dict:
    """
    Reads an uploaded file, determines its type, and calls the appropriate
    parser to extract metadata and a data preview.
    """
    # 1. Read file content ONCE.
    file_content = await file.read()
    size_kb = len(file_content) / 1024
    await file.seek(0)

    data = None
    sample_data = None
    
    # 2. DISPATCHER LOGIC: Decide which parser to use.
    # This logic correctly routes the file to the right parsing function.
    if file.filename.endswith('.csv') or file.filename.endswith('.csv.gz'):
        print(f"Detected CSV file: {file.filename}. Routing to CSV parser.")
        data,sample_data = await parse_csv_file(file_content)

    elif file.filename.endswith('.gz'):
        print(f"Detected .gz file: {file.filename}. Routing to WOD parser.")
        data,sample_data = await parse_wod_file(file_content)

    elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
        print(f"Detected Excel file: {file.filename}. Routing to Excel parser.")
        data,sample_data = await parse_excel_file(file_content)

    elif file.filename.endswith('.parquet'):
        print(f"Detected Parquet file: {file.filename}. Routing to Parquet parser.")
        data,sample_data = await parse_parquet_file(file_content)
    elif file.filename.endswith('.json'):
        print(f"Detected JSON file: {file.filename}. Routing to JSON parser.")
        data, sample_data = await parse_json_file(file_content)

    elif file.filename.endswith('.feather'):
        print(f"Detected Feather file: {file.filename}. Routing to Feather parser.")
        data, sample_data = await parse_feather_file(file_content)

    elif file.filename.endswith(('.h5', '.hdf5')):
        print(f"Detected HDF5 file: {file.filename}. Routing to HDF5 parser.")
        data, sample_data = await parse_hdf5_file(file_content)

    elif file.filename.endswith('.nc'):
        print(f"Detected NetCDF file: {file.filename}. Routing to NetCDF parser.")
        data, sample_data = await parse_netcdf_file(file_content)


    else:
        data = {"error": "Unsupported file type."}

    # 3. Assemble the final response.
    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_kb": round(size_kb, 2),
        "data": data,
        "sample_data": sample_data
    }

    return metadata

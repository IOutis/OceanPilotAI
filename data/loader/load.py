from fastapi import UploadFile
import gzip, shutil
import pandas as pd
import io # Import the io module
from wodpy import wod
import io
import wodpy
import tempfile
import os

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
        profiles_preview = []
        profiles = []
        with open(temp_ascii_path, "r") as fid:
            while True:
                try:
                    profile = wod.WodProfile(fid)
                    variables = []
                    for i in range(len(profile.profile_data)):
                        variables.append(profile.profile_data[i]["variables"])
                    oxygen =profile.oxygen()
                    pH = profile.pH()
                    p = profile.p()
                    z = profile.z()
                    t = profile.t()
                    s = profile.s()
                    silicate = profile.silicate()
                    phosphate = profile.phosphate()
                    
                    profiles.append({
                        "cruise": profile.cruise(),
                        "latitude": float(profile.latitude()),
                        "longitude": float(profile.longitude()),
                        "date": str(profile.datetime()),
                        "levels": int(profile.n_levels()),
                        "depths": z.tolist() if z is not None else [],
                        "biological_header": profile.biological_header,  # if this is a dict, it's safe
                        "oxygen": oxygen.tolist() if oxygen is not None else [],
                        "pH": pH.tolist() if pH is not None else [],
                        "pressure": p.tolist() if p is not None else [],
                        # "nitrate": variables[:3],  # double-check if variables contain arrays too
                        "temperatures":t.tolist() if t is not None else [],
                        "salinity":s.tolist() if s is not None else [],
                        "silicate":silicate.tolist() if silicate is not None else [],
                        "phosphate":phosphate.tolist() if phosphate is not None else [],
                        "var":profile.var_metadata(1)
                    })
                    if len(profiles_preview) < 3:
                        profiles_preview.append({
                            "cruise": profile.cruise(),
                        "latitude": float(profile.latitude()),
                        "longitude": float(profile.longitude()),
                        "date": str(profile.datetime()),
                        "levels": int(profile.n_levels()),
                        "depths": z.tolist()[:3] if z is not None else [],
                        "biological_header": profile.biological_header,  # if this is a dict, it's safe
                        "oxygen": oxygen.tolist()[:3] if oxygen is not None else [],
                        "pH": pH.tolist()[:3] if pH is not None else [],
                        "pressure": p.tolist()[:3] if p is not None else [],
                        # "nitrate": variables[:3],  # double-check if variables contain arrays too
                        "temperatures":t.tolist()[:3] if t is not None else [],
                        "salinity":s.tolist()[:3] if s is not None else [],
                        "silicate":silicate.tolist()[:3] if silicate is not None else [],
                        "phosphate":phosphate.tolist()[:3] if phosphate is not None else [],
                        "var":profile.var_metadata(1)
                        })
                except Exception:
                    break
        
                

        return profiles, profiles_preview

    except Exception as e:
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
        # FIX: Added 'await' to get the result from the async function
        data,sample_data = await parse_csv_file(file_content)
    # A simple check for WOD files which often just have a .gz extension
    elif file.filename.endswith('.gz'):
        print(f"Detected .gz file: {file.filename}. Routing to WOD parser.")
        # FIX: Added 'await' to get the result from the async function
        data,sample_data = await parse_wod_file(file_content)
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

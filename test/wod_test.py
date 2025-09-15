import xarray as xr

def parse_netcdf_file(file_path: str):
    try:
        ds = xr.open_dataset(file_path, engine="netcdf4")  # requires pip install netCDF4
        print("Variables:", list(ds.data_vars))

        # Preview a few variables
        preview = {var: ds[var].values.flatten().tolist() for var in list(ds.data_vars)}
        return preview
    except Exception as e:
        return {"error": str(e)}

res = parse_netcdf_file("mercatorbiomer4v2r1_global_mean_nut_20220131.nc")
print(res)

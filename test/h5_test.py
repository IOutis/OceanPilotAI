import h5py

def explore_h5(path, max_preview=5):
    with h5py.File(path, "r") as f:
        def recurse(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
                # preview some values
                print("  Sample:", obj[tuple(slice(0, min(s, max_preview)) for s in obj.shape)])
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}, keys={list(obj.keys())}")
        
        f.visititems(recurse)

explore_h5("data00000255.h5")

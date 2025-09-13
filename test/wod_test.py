from wodpy import wod

def parse_wod_ascii(file_path, max_profiles=3):
    with open(file_path, 'r') as fid:   # TEXT mode
        for i in range(max_profiles):
            try:
                profile = wod.WodProfile(fid)
                print(f"\n--- Profile {i+1} ---")
                print("Cruise:", profile.cruise())
                for i in range(6):
                    try:
                        print(f"Var at {i} : ",profile.var_metadata(i))
                        print("____________")
                    except Exception:
                        continue
            except EOFError:
                break
parse_wod_ascii("OSDO2022", max_profiles=5)

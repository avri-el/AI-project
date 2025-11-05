import h5py
path = "colon_diseases.h5"  # sesuaikan path

with h5py.File(path, "r") as f:
    print("Root keys:", list(f.keys()))
    # Contoh: tampilkan atribut dan struktur di bawah model_weights
    def print_group(g, indent=0):
        for k, v in g.items():
            print("  " * indent + f"- {k}: {type(v)}")
            if isinstance(v, h5py.Group):
                print_group(v, indent+1)

    print_group(f)
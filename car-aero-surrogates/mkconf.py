# writes one surface benchmark config per model, from NVIDIA's HF matrix config
import yaml, copy, sys
D = "/home/jovyan/aero/physicsnemo-cfd/workflows/benchmarking/conf"
base = yaml.safe_load(open(f"{D}/config_matrix_surface_hf.yaml"))
for m in base["benchmark"]["models"]:
    c = copy.deepcopy(base)
    c["benchmark"]["models"] = [m]
    c["benchmark"]["datasets"][0]["root"] = "/home/jovyan/aero/drivaerml"
    c["run"]["output_dir"] = f"/home/jovyan/aero/out/{m['name']}"
    yaml.safe_dump(c, open(f"{D}/q_{m['name']}.yaml", "w"), sort_keys=False)
    print(m["name"])

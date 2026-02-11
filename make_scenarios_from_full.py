# make_scenarios_from_full.py
import argparse, pandas as pd, re

def parse_scene_dir(scene_dir: str):
    # 例：...\H0_11.0\scene_000__node_J13__t02__seed_12345
    m_h0 = re.search(r"H0_(\d+(?:\.\d+)?)", scene_dir)
    m_sc = re.search(r"scene_(\d+)", scene_dir)
    m_nd = re.search(r"node_(J\d+)", scene_dir)
    m_t  = re.search(r"__t(\d+)", scene_dir)
    m_sd = re.search(r"seed_(\d+)", scene_dir)
    H0 = float(m_h0.group(1)) if m_h0 else None
    scene_id = int(m_sc.group(1)) if m_sc else None
    fault_node = m_nd.group(1) if m_nd else None
    fault_after = int(m_t.group(1)) if m_t else None
    seed = int(m_sd.group(1)) if m_sd else None
    return H0, scene_id, seed, fault_node, fault_after

ap = argparse.ArgumentParser()
ap.add_argument("--full_csv", required=True)
ap.add_argument("--out_csv", required=True)
args = ap.parse_args()

df = pd.read_csv(args.full_csv)

need = ["H0","scene_id","scene_seed","fault_node","fault_after_groups"]
if all(c in df.columns for c in need):
    sc = df[need].copy()
else:
    # fallback: 从 scene_dir 解析
    if "scene_dir" not in df.columns:
        raise RuntimeError("full_csv missing required columns and scene_dir.")
    rows = []
    for sd in df["scene_dir"].astype(str).tolist():
        H0, scene_id, seed, fault_node, fault_after = parse_scene_dir(sd)
        rows.append([H0, scene_id, seed, fault_node, fault_after])
    sc = pd.DataFrame(rows, columns=need)

sc = sc.dropna().drop_duplicates()
sc["H0"] = sc["H0"].astype(float).round(1)
sc["scene_id"] = sc["scene_id"].astype(int)
sc["scene_seed"] = sc["scene_seed"].astype("int64")
sc["fault_after_groups"] = sc["fault_after_groups"].astype(int)
sc = sc.sort_values(["H0","scene_id"]).reset_index(drop=True)
sc.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
print("WROTE:", args.out_csv, "rows=", len(sc))

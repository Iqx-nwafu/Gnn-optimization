
# batch_infer_fault.py
# Batch wrapper for fault-aware inference:
#   Loop H0 in [start,end] with step; for each H0 sample K random (fault_node, fault_after_groups) scenarios
#   Run infer_gnn_generator_fault.py for each scenario (subprocess)
#   Aggregate summary CSV across all runs (and per-H0 CSV)
#
# Reproducibility:
#   scenario_seed = stable_hash(base_seed, H0, scenario_id)  (deterministic; independent of execution order)
#
# Output structure:
#   out_root/
#     H0_17.8/
#       scene_000__node_J12__t08__seed_123456789/
#         schedule.json, group_metrics.csv, lateral_metrics.csv, summary.json
#       ...
#     summary_all.csv
#
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# needs tree_evaluator to enumerate nodes
import tree_evaluator as te


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # core inputs
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--generator_ckpt", required=True)
    ap.add_argument("--feas_ckpt", default="")

    ap.add_argument("--nodes", required=True)
    ap.add_argument("--pipes", required=True)
    ap.add_argument("--root", default="J0")
    ap.add_argument("--Hmin", type=float, default=11.59)
    ap.add_argument("--q_lateral", type=float, default=0.012)

    # H0 sweep
    ap.add_argument("--H0_start", type=float, default=11.0)
    ap.add_argument("--H0_end", type=float, default=23.0)
    ap.add_argument("--H0_step", type=float, default=0.2)

    # scenarios
    ap.add_argument("--scenes_per_H0", type=int, default=50)
    ap.add_argument("--prefault_force_k", type=int, default=4, help="engineering default: 4 before fault")
    ap.add_argument("--max_prefault_groups", type=int, default=30, help="used to sample fault_time; 30 for 120 laterals / 4 per group")
    ap.add_argument("--base_seed", type=int, default=123)
    ap.add_argument("--fault_nodes_mode", choices=["field_only", "all"], default="field_only")

    # runtime
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top_k", type=int, default=14)
    ap.add_argument("--max_expand", type=int, default=24)
    ap.add_argument("--feas_threshold", type=float, default=0.35)
    ap.add_argument("--max_random_tries", type=int, default=800)

    # output
    ap.add_argument("--infer_script", default="infer_gnn_generator_fault.py", help="path to infer script")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--skip_existing", action="store_true", help="resume: skip scenes with existing schedule.json")
    ap.add_argument("--fail_fast", action="store_true", help="stop immediately on first failed subprocess")

    return ap.parse_args()


def stable_int_hash(s: str) -> int:
    """
    Simple stable hash (FNV-1a 32-bit) so results do not depend on Python's hash randomization.
    """
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def scenario_seed(base_seed: int, H0: float, scene_id: int) -> int:
    # represent H0 to 1 decimal to avoid float issues (0.2 step -> 1 decimal sufficient)
    key = f"{base_seed}|{H0:.1f}|{scene_id}"
    return stable_int_hash(key)


def enumerate_H0(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("H0_step must be > 0")
    n = int(round((end - start) / step)) + 1
    vals = []
    for i in range(n):
        v = start + i * step
        # round to 1 decimal for 0.2-step grid; keep safe
        vals.append(round(v, 1))
    # ensure end included (within tolerance)
    if abs(vals[-1] - end) > 1e-6:
        # adjust by clipping
        vals[-1] = round(end, 1)
    return vals


def is_mainline_node_id(nid: str) -> bool:
    m = re.fullmatch(r"J(\d+)", str(nid).strip())
    if not m:
        return False
    return int(m.group(1)) <= 10


def choose_fault_nodes(nodes_xlsx: str, mode: str) -> List[str]:
    nodes = te.load_nodes_xlsx(nodes_xlsx)
    ids = list(nodes.keys())
    if mode == "all":
        # exclude root
        out = [nid for nid in ids if nid != "J0" and not is_mainline_node_id(nid)]
        # keep only Jxx-like
        out = [nid for nid in out if re.fullmatch(r"J\d+", nid)]
        return sorted(out, key=lambda x: int(x[1:]))
    # field_only: use helper when available
    out = [nid for nid in ids if te.is_field_node_id(nid)]
    # enforce "mainline not fail"
    out = [nid for nid in out if not is_mainline_node_id(nid)]
    # stable sort by numeric id
    out = [nid for nid in out if re.fullmatch(r"J\d+", nid)]
    out = sorted(out, key=lambda x: int(x[1:]))
    if not out:
        raise RuntimeError("No candidate fault nodes found (field_only). Check Nodes.xlsx and te.is_field_node_id().")
    return out


def run_scene(args: argparse.Namespace, H0: float, scene_id: int, fault_node: str, fault_after_groups: int, seed: int, out_dir: Path) -> Tuple[bool, str]:
    """
    Launch infer script as subprocess.
    Returns (ok, message). On success, message is stdout tail; on failure, message is stderr tail.
    """
    infer = Path(args.infer_script)
    if not infer.exists():
        # allow relative to this script folder
        here = Path(__file__).resolve().parent
        infer2 = here / args.infer_script
        if infer2.exists():
            infer = infer2
        else:
            raise FileNotFoundError(f"infer_script not found: {args.infer_script}")

    cmd = [
        sys.executable, str(infer),
        "--dataset_dir", args.dataset_dir,
        "--generator_ckpt", args.generator_ckpt,
        "--nodes", args.nodes,
        "--pipes", args.pipes,
        "--root", args.root,
        "--Hmin", str(args.Hmin),
        "--q_lateral", str(args.q_lateral),
        "--H0", f"{H0:.1f}",
        "--fault_node", fault_node,
        "--fault_after_groups", str(fault_after_groups),
        "--prefault_force_k", str(args.prefault_force_k),
        "--out", str(out_dir),
        "--top_k", str(args.top_k),
        "--max_expand", str(args.max_expand),
        "--feas_threshold", str(args.feas_threshold),
        "--max_random_tries", str(args.max_random_tries),
        "--device", args.device,
        "--seed", str(seed),
    ]
    if args.feas_ckpt.strip():
        cmd += ["--feas_ckpt", args.feas_ckpt]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 0:
        msg = (p.stdout or "").strip().splitlines()[-3:]
        return True, " | ".join(msg) if msg else "ok"
    else:
        err = (p.stderr or p.stdout or "").strip().splitlines()[-8:]
        return False, " | ".join(err) if err else f"returncode={p.returncode}"


def load_scene_summary(scene_dir: Path) -> Dict:
    sj = scene_dir / "schedule.json"
    if not sj.exists():
        raise FileNotFoundError(str(sj))
    return json.loads(sj.read_text(encoding="utf-8"))


def append_csv(path: Path, header: List[str], rows: List[List[object]], write_header_if_new: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header_if_new and not exists:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # total laterals from dataset meta (avoid hardcoding 120)
    meta_path = Path(args.dataset_dir) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        total_laterals = int(len(meta.get("lateral_ids", [])))
    else:
        total_laterals = 120  # fallback (your design)
    if total_laterals <= 0:
        total_laterals = 120

    H0_list = enumerate_H0(args.H0_start, args.H0_end, args.H0_step)
    fault_nodes = choose_fault_nodes(args.nodes, args.fault_nodes_mode)

    # try to infer prefault max groups from laterals count (if you want), but keep args.max_prefault_groups as default
    # here we stick to args.max_prefault_groups so it's explicit and reproducible.
    max_t = max(0, int(args.max_prefault_groups) - 1)

    summary_all = out_root / "summary_all.csv"
    header = [
        "H0", "scene_id", "scene_seed",
        "fault_node", "fault_after_groups",
        "status", "message",
        "n_groups_total", "n_groups_postfault",
        "J_all", "mu_all", "std_all",
        "J_post", "mu_post", "std_post",
        "blocked_laterals", "lost_laterals", "lost_ratio",
        "scene_dir",
    ]

    total = 0
    failed = 0

    for H0 in H0_list:
        # per-H0 CSV
        summary_h0 = out_root / f"H0_{H0:.1f}" / "summary.csv"
        for scene_id in range(args.scenes_per_H0):
            total += 1
            sseed = scenario_seed(args.base_seed, H0, scene_id)
            rng = random_from_seed(sseed)

            fault_node = rng.choice(fault_nodes)
            fault_after_groups = rng.randint(0, max_t)  # inclusive

            scene_dir = out_root / f"H0_{H0:.1f}" / f"scene_{scene_id:03d}__node_{fault_node}__t{fault_after_groups:02d}__seed_{sseed}"
            sj = scene_dir / "schedule.json"
            if args.skip_existing and sj.exists():
                # still record from existing outputs
                try:
                    payload = load_scene_summary(scene_dir)
                    row = row_from_payload(H0, scene_id, sseed, fault_node, fault_after_groups, "SKIP_OK", "exists", payload, scene_dir, total_laterals)
                except Exception as e:
                    failed += 1
                    row = [f"{H0:.1f}", scene_id, sseed, fault_node, fault_after_groups, "SKIP_FAIL", repr(e),
                           "", "", "", "", "", "", "", "", "", "", "", str(scene_dir)]
                append_csv(summary_all, header, [row])
                append_csv(summary_h0, header, [row])
                continue

            ok, msg = run_scene(args, H0, scene_id, fault_node, fault_after_groups, sseed, scene_dir)
            if not ok:
                failed += 1
                row = [f"{H0:.1f}", scene_id, sseed, fault_node, fault_after_groups, "FAIL", msg,
                       "", "", "", "", "", "", "", "", "", "", "", str(scene_dir)]
                append_csv(summary_all, header, [row])
                append_csv(summary_h0, header, [row])
                print(f"[FAIL] H0={H0:.1f} scene={scene_id} node={fault_node} t={fault_after_groups} :: {msg}")
                if args.fail_fast:
                    raise SystemExit(1)
                continue

            # load outputs
            try:
                payload = load_scene_summary(scene_dir)
                row = row_from_payload(H0, scene_id, sseed, fault_node, fault_after_groups, "OK", msg, payload, scene_dir, total_laterals)
            except Exception as e:
                failed += 1
                row = [f"{H0:.1f}", scene_id, sseed, fault_node, fault_after_groups, "POST_FAIL", repr(e),
                       "", "", "", "", "", "", "", "", "", "", "", str(scene_dir)]

            append_csv(summary_all, header, [row])
            append_csv(summary_h0, header, [row])

            print(f"[OK] H0={H0:.1f} scene={scene_id:03d} node={fault_node} t={fault_after_groups:02d} seed={sseed} :: {msg}")

    print(f"[DONE] total={total} failed={failed} out={out_root}")


def random_from_seed(seed: int):
    import random
    r = random.Random()
    r.seed(int(seed) & 0xFFFFFFFF)
    return r


def safe_float(x) -> str:
    try:
        if x is None:
            return ""
        return f"{float(x):.6g}"
    except Exception:
        return ""


def row_from_payload(H0: float, scene_id: int, sseed: int, fault_node: str, fault_after_groups: int,
                     status: str, message: str, payload: Dict, scene_dir: Path) -> List[object]:
    glb = payload.get("global", {}) or {}
    post = (glb.get("postfault", {}) or {})
    fault = payload.get("fault", {}) or {}

    n_groups_total = glb.get("n_groups", "")
    n_groups_post = post.get("n_groups", "")

    J_all = glb.get("J", "")
    mu_all = glb.get("mu_group_mean", "")
    std_all = glb.get("std_group_mean", "")

    J_post = post.get("J", "")
    mu_post = post.get("mu_group_mean", "")
    std_post = post.get("std_group_mean", "")

    blocked = fault.get("blocked_laterals", []) or []
    lost = fault.get("lost_laterals", []) or []
    try:
        lost_ratio = float(len(lost)) / max(1, len(payload.get("groups", [])) * 0 + 120)  # 120 laterals by design
    except Exception:
        lost_ratio = ""

    return [
        f"{H0:.1f}", scene_id, sseed,
        fault_node, fault_after_groups,
        status, message,
        n_groups_total, n_groups_post,
        safe_float(J_all), safe_float(mu_all), safe_float(std_all),
        safe_float(J_post), safe_float(mu_post), safe_float(std_post),
        len(blocked), len(lost), safe_float(lost_ratio),
        str(scene_dir),
    ]


if __name__ == "__main__":
    main()

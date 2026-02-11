#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_fault_summary_all_v2.py

Why you saw many empty columns:
- The v1 collector only read fields from payload.json. If payload.json doesn't include H0/fault/status/runtime/etc,
  those columns become empty.
- Your payload keys use names like `mu_group_mean`, `std_group_mean`, `postfault.mu_group_mean`, etc.
  v1 created canonical columns (`mu_all`, `std_all`, ...) but didn't map these variants -> canonical, so canonical
  columns stayed empty while the raw columns existed, creating many sparse columns.

What this v2 does:
- Derives H0 / fault_node / fault_after_groups / scene_id / scene_seed from the scene folder name and path:
    ...\H0_11.0\scene_000__node_J13__t02__seed_123000369\
- Reads runtime_sec from runtime_sec.txt if not present in payload.json.
- Maps common variant keys (mu_group_mean, std_group_mean, J, postfault.*) into canonical columns.
- Drops columns that are entirely empty by default (you can keep them with --keep_all_empty_cols).

Usage:
  python collect_fault_summary_all_v2.py --runs_root "E:\\...\\runs_fault_infer_full" --out_csv "E:\\...\\summary_all.csv"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import pandas as pd


DEFAULT_PAYLOAD_NAMES = [
    "payload.json",
    "scene_payload.json",
    "result.json",
    "summary.json",
    "metrics.json",
]


RE_H0 = re.compile(r"(?:^|[\\/])H0[_=](?P<h0>\d+(?:\.\d+)?)", re.IGNORECASE)
RE_SCENE = re.compile(
    r"scene_(?P<scene>\d+)"
    r"(?:__node_(?P<node>[^_]+))?"
    r"(?:__t(?P<t>\d+))?"
    r"(?:__seed_(?P<seed>\d+))?",
    re.IGNORECASE,
)

def flatten_dict(d: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            flatten_dict(v, kk, out)
    elif isinstance(d, list):
        # Keep short lists as JSON; long lists as length only
        if len(d) <= 20:
            out[prefix] = json.dumps(d, ensure_ascii=False)
        else:
            out[prefix] = len(d)
    else:
        out[prefix] = d
    return out

def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return int(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def pick_first(flat: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in flat and flat[k] not in (None, "", "null", "None"):
            return flat[k]
    return None

def parse_meta_from_path(scene_dir: Path) -> Dict[str, Any]:
    s = str(scene_dir)
    meta: Dict[str, Any] = {}
    m = RE_H0.search(s)
    if m:
        meta["H0"] = as_float(m.group("h0"))
    m2 = RE_SCENE.search(scene_dir.name)
    if m2:
        meta["scene_id"] = as_int(m2.group("scene"))
        if m2.group("node"):
            meta["fault_node"] = m2.group("node")
        if m2.group("t"):
            meta["fault_after_groups"] = as_int(m2.group("t"))
        if m2.group("seed"):
            meta["scene_seed"] = as_int(m2.group("seed"))
    return meta

def find_payload(scene_dir: Path, payload_names: List[str]) -> Optional[Path]:
    for nm in payload_names:
        p = scene_dir / nm
        if p.exists():
            return p
    # fallback: first json
    js = sorted(scene_dir.glob("*.json"))
    return js[0] if js else None

def read_runtime_sec(scene_dir: Path) -> Optional[float]:
    p = scene_dir / "runtime_sec.txt"
    if p.exists():
        return as_float(p.read_text(encoding="utf-8", errors="ignore").strip())
    return None

def read_evaluator_calls(scene_dir: Path) -> Optional[int]:
    p = scene_dir / "evaluator_calls.txt"
    if p.exists():
        return as_int(p.read_text(encoding="utf-8", errors="ignore").strip())
    return None

def canonicalize(flat: Dict[str, Any]) -> Dict[str, Any]:
    # create canonical columns; keep existing keys too
    def set_if_missing(dst_key: str, candidates: List[str]) -> None:
        if flat.get(dst_key) in (None, "", "null", "None"):
            v = pick_first(flat, candidates)
            if v is not None:
                flat[dst_key] = v

    # groups / objective
    set_if_missing("n_groups_total", ["n_groups_total", "n_groups", "n_groups_all", "all.n_groups"])
    set_if_missing("J_all", ["J_all", "J", "objective", "obj", "all.J", "all.obj"])
    set_if_missing("mu_all", ["mu_all", "mu_group_mean", "mu_mean", "group.mu_mean", "all.mu_group_mean", "all.mu_mean"])
    set_if_missing("std_all", ["std_all", "std_group_mean", "sigma_group_mean", "std_mean", "group.std_mean", "all.std_group_mean"])

    # postfault
    set_if_missing("n_groups_postfault", ["n_groups_postfault", "postfault.n_groups", "postfault.n_groups_total", "post.n_groups"])
    set_if_missing("J_post", ["J_post", "postfault.J", "postfault.obj", "post.J"])
    set_if_missing("mu_post", ["mu_post", "postfault.mu_group_mean", "postfault.mu_mean", "post.mu_group_mean", "post.mu_mean"])
    set_if_missing("std_post", ["std_post", "postfault.std_group_mean", "postfault.std_mean", "post.std_group_mean", "post.std_mean"])

    # lost / blocked
    set_if_missing("blocked_laterals", ["blocked_laterals", "blocked", "blocked_count", "loss.blocked"])
    set_if_missing("lost_laterals", ["lost_laterals", "lost", "lost_count", "loss.lost"])
    set_if_missing("lost_ratio", ["lost_ratio", "loss_ratio", "lostRate", "lost_rate"])

    # coverage
    set_if_missing("reachable_total", ["reachable_total", "reachable_laterals", "reach_total"])
    set_if_missing("irrigated_total", ["irrigated_total", "irrigated_laterals", "irrig_total"])
    set_if_missing("coverage", ["coverage", "coverage_ratio", "cov"])

    # runtime / eval calls
    set_if_missing("runtime_sec", ["runtime_sec", "runtime", "time_sec", "elapsed_sec"])
    set_if_missing("evaluator_calls", ["evaluator_calls", "eval_calls", "hydraulic_calls"])

    # if coverage missing, derive
    if flat.get("coverage") in (None, "", "null", "None"):
        rt = as_float(flat.get("reachable_total"))
        it = as_float(flat.get("irrigated_total"))
        if rt and it is not None and rt > 0:
            flat["coverage"] = it / rt

    return flat

def iter_scene_dirs(runs_root: Path) -> Iterable[Path]:
    # scene dirs are leaf folders that contain payload json OR have name starting with 'scene_'
    for p in runs_root.rglob("*"):
        if not p.is_dir():
            continue
        if p.name.lower().startswith("scene_"):
            yield p

def collect_rows(runs_root: Path, payload_names: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene_dir in iter_scene_dirs(runs_root):
        payload_path = find_payload(scene_dir, payload_names)
        flat: Dict[str, Any] = {}
        if payload_path and payload_path.exists():
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                flat = flatten_dict(payload)
            except Exception as e:
                flat = {"status": "PAYLOAD_READ_FAIL", "message": repr(e)}
        else:
            flat = {"status": "NO_PAYLOAD", "message": ""}

        # meta from path
        meta = parse_meta_from_path(scene_dir)
        for k, v in meta.items():
            flat.setdefault(k, v)

        flat.setdefault("scene_dir", str(scene_dir))
        if payload_path:
            flat.setdefault("payload_file", str(payload_path))

        # runtime/eval calls from sidecar files if missing
        if flat.get("runtime_sec") in (None, "", "null", "None"):
            rt = read_runtime_sec(scene_dir)
            if rt is not None:
                flat["runtime_sec"] = rt
        if flat.get("evaluator_calls") in (None, "", "null", "None"):
            ec = read_evaluator_calls(scene_dir)
            if ec is not None:
                flat["evaluator_calls"] = ec

        flat = canonicalize(flat)
        rows.append(flat)
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--payload_names", default=",".join(DEFAULT_PAYLOAD_NAMES))
    ap.add_argument("--keep_all_empty_cols", action="store_true")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_csv = Path(args.out_csv)
    payload_names = [x.strip() for x in args.payload_names.split(",") if x.strip()]

    rows = collect_rows(runs_root, payload_names)
    if not rows:
        raise SystemExit(f"No scene folders found under {runs_root}")

    df = pd.DataFrame(rows)

    # put important columns first if present
    first_cols = [
        "H0","scene_id","scene_seed","fault_node","fault_after_groups",
        "status","message",
        "reachable_total","irrigated_total","coverage",
        "n_groups_total","n_groups_postfault",
        "J_all","mu_all","std_all",
        "J_post","mu_post","std_post",
        "blocked_laterals","lost_laterals","lost_ratio",
        "evaluator_calls","runtime_sec","feas_threshold",
        "scene_dir","payload_file"
    ]
    ordered = [c for c in first_cols if c in df.columns] + [c for c in df.columns if c not in first_cols]
    df = df[ordered]

    if not args.keep_all_empty_cols:
        # drop columns that are entirely empty
        df = df.dropna(axis=1, how="all")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[DONE] wrote:", out_csv, "rows:", len(df), "cols:", df.shape[1])

if __name__ == "__main__":
    main()

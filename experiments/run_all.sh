#!/bin/bash
# Full evaluation pipeline after training (see README.md). Run on an otherwise idle 4-core CPU:
# every method is timed on a single CPU thread, scenarios are spread over 4 worker processes.
set -e
cd "$(dirname "$0")"
python3 bench_id.py        > log_bench_id.txt 2>&1     # res_id.json
python3 bench_ood.py       > log_bench_ood.txt 2>&1    # res_ood.json
python3 pareto.py          > log_pareto.txt 2>&1       # res_pareto.json
python3 final_eval.py      > log_final_eval.txt 2>&1   # res_final.json
python3 test_split_eval.py > log_testsplit.txt 2>&1    # res_testsplit.json
python3 ood_final.py       > log_ood_final.txt 2>&1    # res_ood_final.json
python3 ood_blockage.py    > log_blockage.txt 2>&1     # res_blockage.json
python3 pareto_gnn.py      > log_pareto_gnn.txt 2>&1   # res_pareto_gnn.json
python3 epa_final.py       > log_epa_final.txt 2>&1    # res_final_epa.json
python3 margin_eval.py     > log_margin.txt 2>&1       # res_margin.json
python3 full_net.py        > log_fullnet.txt 2>&1      # res_fullnet.json
python3 make_tables.py
python3 figs.py
echo ALLDONE

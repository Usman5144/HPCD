from __future__ import annotations
import os

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / 'code/hpcd'))
from hpcd_controller_v1 import HJTMModel
from reanalyze_t2b7_common_metrics_v1 import score_tokens
INPUT = ROOT / 'results/hpcd/t2c4_dev111_surface_entity_selection_v2/selected_outputs.jsonl'
EXPECTED_INPUT_SHA = '4676157d02d323fd1f78d0a97e172c73caf81a43d4887aaa07b09652fa2ca3dd'
MODEL_PATH = Path('/path/to/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1')
HJTM_PATH = ROOT / 'results/hjtm/final_hjtm_v1/hjtm_final_v1.json'
OUT = ROOT / 'results/hpcd/t2c4_dev111_common_rescore_v1.json'
CONDITIONS = ('vanilla', 'hpcd_n0', 'hpcd_n1')
ORDERS = (0, 1)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_jsonl(path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def mean(values):
    if not values:
        return None
    return float(np.mean(values))

def median(values):
    if not values:
        return None
    return float(np.median(values))

def bootstrap_ci(values, *, seed=2027, B=10000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]

def summarize_delta(name, deltas):
    deltas = [float(x) for x in deltas if math.isfinite(float(x))]
    result = {'n': len(deltas), 'mean_delta': mean(deltas), 'median_delta': median(deltas), 'improved_negative': int(sum((x < 0 for x in deltas))), 'ties': int(sum((x == 0 for x in deltas))), 'worsened_positive': int(sum((x > 0 for x in deltas))), 'bootstrap_95_ci_mean': bootstrap_ci(deltas)}
    print()
    print(name)
    print('  n:', result['n'])
    print('  mean delta:', f"{result['mean_delta']:+.6f}")
    print('  median delta:', f"{result['median_delta']:+.6f}")
    print('  improved (<0):', f"{result['improved_negative']}/{result['n']}")
    print('  worsened (>0):', f"{result['worsened_positive']}/{result['n']}")
    print('  95% bootstrap CI:', result['bootstrap_95_ci_mean'])
    return result

def main():
    print('=' * 100)
    print('T2C.3 — COMMON HJTM RESCORING')
    print('=' * 100)
    input_sha = sha256_file(INPUT)
    print('selected input SHA256:', input_sha)
    if input_sha != EXPECTED_INPUT_SHA:
        raise RuntimeError('Corrected selected-output hash mismatch')
    if OUT.exists():
        raise RuntimeError(f'Refusing overwrite: {OUT}')
    rows = load_jsonl(INPUT)
    if len(rows) != 999:
        raise RuntimeError(f'Expected 72 rows; found {len(rows)}')
    instances = sorted({r['instance_id'] for r in rows})
    if len(instances) != 333:
        raise RuntimeError(f'Expected 24 documents; found {len(instances)}')
    row_map = {(r['instance_id'], r['condition']): r for r in rows}
    if len(row_map) != 999:
        raise RuntimeError('Duplicate instance/condition row')
    print('documents:', len(instances))
    print('rows:', len(rows))
    if not MODEL_PATH.exists():
        raise RuntimeError(f'Missing model: {MODEL_PATH}')
    if not HJTM_PATH.exists():
        raise RuntimeError(f'Missing HJTM: {HJTM_PATH}')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('device:', device)
    print()
    print('Loading frozen Qwen2.5-3B reference model...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    hjtm = HJTMModel(HJTM_PATH)
    unique = {}
    for row in rows:
        key = (row['domain'], row['output_text'])
        unique.setdefault(key, None)
    print('unique domain/text outputs:', len(unique))
    score_cache = {}
    total = len(unique) * len(ORDERS)
    done = 0
    for domain, text in unique:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 2:
            raise RuntimeError('Output tokenized to fewer than 2 tokens')
        for order in ORDERS:
            result = score_tokens(token_ids, model=model, hjtm=hjtm, device=device, domain=domain, order=order, eos_token_id=tokenizer.eos_token_id)
            score_cache[domain, text, order] = result
            done += 1
            print(f"[{done:03d}/{total:03d}] domain={domain:7s} n={order} tokens={result['scored_tokens']:4d} NLL={result['nll']:.6f} T={result['typicality']:.6f}", flush=True)
    rescored = []
    for row in rows:
        rec = dict(row)
        for order in ORDERS:
            result = score_cache[row['domain'], row['output_text'], order]
            rec[f'common_nll_n{order}'] = float(result['nll'])
            rec[f'common_typicality_n{order}'] = float(result['typicality'])
            rec[f'common_scored_tokens_n{order}'] = int(result['scored_tokens'])
        rescored.append(rec)
    rmap = {(r['instance_id'], r['condition']): r for r in rescored}
    matched_n0_n1 = [iid for iid in instances if rmap[iid, 'hpcd_n0']['status'] == 'REWRITE' and rmap[iid, 'hpcd_n1']['status'] == 'REWRITE']
    matched_all = [iid for iid in instances if all((rmap[iid, condition]['status'] == 'REWRITE' for condition in CONDITIONS))]
    print()
    print('=' * 100)
    print('MATCHED SETS')
    print('=' * 100)
    print('n0+n1 rewrites:', f'{len(matched_n0_n1)}/24')
    print('Vanilla+n0+n1 rewrites:', f'{len(matched_all)}/24')
    print()
    print('=' * 100)
    print('CENTRAL CAUSAL TEST: HPCD n1 vs HPCD n0')
    print('NEGATIVE DELTA = n1 OUTPUT IS MORE HUMAN-TYPICAL')
    print('=' * 100)
    central = {}
    for score_order in ORDERS:
        deltas = []
        for iid in matched_n0_n1:
            n0 = rmap[iid, 'hpcd_n0']
            n1 = rmap[iid, 'hpcd_n1']
            delta = n1[f'common_typicality_n{score_order}'] - n0[f'common_typicality_n{score_order}']
            deltas.append(delta)
        central[f'n1_minus_n0_scored_n{score_order}'] = summarize_delta(f'n1 - n0, COMMON SCORE n={score_order}', deltas)
    print()
    print('=' * 100)
    print('MATCHED COMPARISON AGAINST VANILLA')
    print('NEGATIVE DELTA = HPCD IS MORE HUMAN-TYPICAL')
    print('=' * 100)
    versus_vanilla = {}
    comparisons = (('hpcd_n0', 0), ('hpcd_n1', 1), ('hpcd_n0', 1), ('hpcd_n1', 0))
    for condition, score_order in comparisons:
        deltas = []
        for iid in matched_all:
            vanilla = rmap[iid, 'vanilla']
            hpcd = rmap[iid, condition]
            delta = hpcd[f'common_typicality_n{score_order}'] - vanilla[f'common_typicality_n{score_order}']
            deltas.append(delta)
        key = f'{condition}_minus_vanilla_scored_n{score_order}'
        versus_vanilla[key] = summarize_delta(key, deltas)
    sensitivity_ids = [iid for iid in matched_n0_n1 if float(rmap[iid, 'hpcd_n0']['length_ratio']) >= 0.8 and float(rmap[iid, 'hpcd_n1']['length_ratio']) >= 0.8]
    print()
    print('=' * 100)
    print('POST-HOC LENGTH >= 0.80 SENSITIVITY')
    print('DIAGNOSTIC ONLY — NOT A FROZEN QUALITY RULE')
    print('=' * 100)
    print('matched n0+n1 retained:', f'{len(sensitivity_ids)}/{len(matched_n0_n1)}')
    sensitivity = {}
    for score_order in ORDERS:
        deltas = []
        for iid in sensitivity_ids:
            n0 = rmap[iid, 'hpcd_n0']
            n1 = rmap[iid, 'hpcd_n1']
            deltas.append(n1[f'common_typicality_n{score_order}'] - n0[f'common_typicality_n{score_order}'])
        sensitivity[f'n1_minus_n0_scored_n{score_order}'] = summarize_delta(f'length>=0.80: n1-n0 scored n={score_order}', deltas)
    print()
    print('=' * 100)
    print('PER-DOCUMENT n0/n1 COMMON RESCORE')
    print('=' * 100)
    pair_rows = []
    for iid in matched_n0_n1:
        n0 = rmap[iid, 'hpcd_n0']
        n1 = rmap[iid, 'hpcd_n1']
        pair = {'instance_id': iid, 'domain': n0['domain'], 'n0_length_ratio': n0['length_ratio'], 'n1_length_ratio': n1['length_ratio'], 'n0_T0': n0['common_typicality_n0'], 'n1_T0': n1['common_typicality_n0'], 'delta_T0_n1_minus_n0': n1['common_typicality_n0'] - n0['common_typicality_n0'], 'n0_T1': n0['common_typicality_n1'], 'n1_T1': n1['common_typicality_n1'], 'delta_T1_n1_minus_n0': n1['common_typicality_n1'] - n0['common_typicality_n1']}
        pair_rows.append(pair)
        print(f"{iid} | {pair['domain']:7s} | T0: {pair['n0_T0']:.4f} -> {pair['n1_T0']:.4f} d={pair['delta_T0_n1_minus_n0']:+.4f} | T1: {pair['n0_T1']:.4f} -> {pair['n1_T1']:.4f} d={pair['delta_T1_n1_minus_n0']:+.4f}")
    output = {'status': 'COMPLETE', 'stage': 'T2C.3 exact common HJTM rescoring', 'input_sha256': input_sha, 'scorer': 'score_tokens imported directly from reanalyze_t2b7_common_metrics_v1.py', 'orders': list(ORDERS), 'interpretation': 'For paired deltas, negative values mean the first named HPCD output is closer to the human typicality target under the same common scorer.', 'matched_n0_n1_ids': matched_n0_n1, 'matched_all_three_ids': matched_all, 'central_n1_vs_n0': central, 'versus_vanilla': versus_vanilla, 'posthoc_length_0_80_sensitivity': {'note': 'Diagnostic only; 0.80 was not used as a frozen selection rule.', 'ids': sensitivity_ids, 'comparisons': sensitivity}, 'paired_rows': pair_rows, 'rescored_selected_outputs': rescored}
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print()
    print('=' * 100)
    print('COMMON RESCORING COMPLETE')
    print('=' * 100)
    print('output SHA256:', sha256_file(OUT))
    print('script SHA256:', sha256_file(Path(__file__).resolve()))
if __name__ == '__main__':
    main()

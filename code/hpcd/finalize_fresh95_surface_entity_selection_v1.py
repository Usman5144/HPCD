from __future__ import annotations
import os

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import spacy
ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
INPUT = ROOT / 'results/hpcd/fresh95_quality_stage1_v1/scored_candidates.jsonl'
EXPECTED_SHA = '32ac20ec25c50aebbdc10a2ecb72b3a7a2a9beb3f1e39508acb0c460f8383335'
OUT_DIR = ROOT / 'results/hpcd/fresh95_surface_entity_selection_v1'
SCORED_OUT = OUT_DIR / 'rescored_candidates.jsonl'
SELECTED_OUT = OUT_DIR / 'selected_outputs.jsonl'
SUMMARY_OUT = OUT_DIR / 'summary.json'
CONDITIONS = ('vanilla_t10', 'hpcd_n0', 'vanilla_t11')
K = 4
ATTEMPTS = 2

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_jsonl(path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def normalize_surface(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').replace('–', '-').replace('—', '-').casefold()
    return re.sub('\\s+', ' ', text).strip()

def source_entities(nlp, text):
    ignore = {'CARDINAL', 'ORDINAL', 'PERCENT', 'MONEY', 'DATE', 'TIME', 'QUANTITY'}
    entities = []
    seen = set()
    for ent in nlp(text).ents:
        if ent.label_ in ignore:
            continue
        value = normalize_surface(ent.text)
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            entities.append(value)
    return entities

def entity_present(entity, normalized_output):
    pattern = '(?<!\\w)' + re.escape(entity) + '(?!\\w)'
    return re.search(pattern, normalized_output) is not None

def surface_entity_check(entities, output):
    normalized_output = normalize_surface(output)
    missing = [entity for entity in entities if not entity_present(entity, normalized_output)]
    return (len(missing) == 0, missing)

def main():
    print('=' * 100)
    print('T2C.3 — CORRECTED SOURCE-ENTITY SURFACE GATE')
    print('=' * 100)
    actual_sha = sha256_file(INPUT)
    print('input SHA256:', actual_sha)
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError('Frozen scored candidate hash mismatch')
    rows = load_jsonl(INPUT)
    if len(rows) != 2280:
        raise RuntimeError(f'Expected 2280 candidates, got {len(rows)}')
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    nlp = spacy.load('en_core_web_sm')
    source_cache = {}
    rescored = []
    for row in rows:
        iid = row['instance_id']
        if iid not in source_cache:
            source_cache[iid] = source_entities(nlp, row['source_text'])
        ok, missing = surface_entity_check(source_cache[iid], row['text'])
        rec = dict(row)
        rec['source_entity_surfaces'] = source_cache[iid]
        rec['missing_source_entity_surfaces'] = missing
        rec['surface_entity_ok'] = bool(ok)
        rec['quality_valid_v2'] = bool(row['semantic_ok'] and row['number_ok'] and row['date_ok'] and ok and row['text'].strip())
        rescored.append(rec)
    write_jsonl(SCORED_OUT, rescored)
    print()
    print('=' * 100)
    print('OLD vs CORRECTED CANDIDATE VALIDITY')
    print('=' * 100)
    for condition in CONDITIONS:
        subset = [r for r in rescored if r['condition'] == condition]
        old_valid = sum((r['quality_valid'] for r in subset))
        new_valid = sum((r['quality_valid_v2'] for r in subset))
        old_entity_fail = sum((not r['entity_ok'] for r in subset))
        new_entity_fail = sum((not r['surface_entity_ok'] for r in subset))
        print()
        print(condition)
        print(f'  old valid: {old_valid}/192 ({old_valid / 760:.1%})')
        print(f'  new valid: {new_valid}/192 ({new_valid / 760:.1%})')
        print(f'  old entity failures: {old_entity_fail}/192')
        print(f'  corrected entity failures: {new_entity_fail}/192')
    print()
    print('=' * 100)
    print('CORRECTED VALIDITY BY DOMAIN')
    print('=' * 100)
    for domain in ('news', 'reddit', 'reviews', 'wiki'):
        print()
        print(domain.upper())
        for condition in CONDITIONS:
            subset = [r for r in rescored if r['domain'] == domain and r['condition'] == condition]
            valid = sum((r['quality_valid_v2'] for r in subset))
            entity_fail = sum((not r['surface_entity_ok'] for r in subset))
            print(f'  {condition:8s} valid={valid:2d}/48 ({valid / 48:.1%}) surface_entity_fail={entity_fail:2d}')
    grouped = defaultdict(list)
    for row in rescored:
        grouped[row['instance_id'], row['condition'], int(row['slot'])].append(row)
    slot_valid = defaultdict(list)
    for key, attempts in grouped.items():
        attempts = sorted(attempts, key=lambda r: int(r['attempt']))
        if len(attempts) != ATTEMPTS:
            raise RuntimeError(f'Expected 2 attempts: {key}')
        chosen = next((r for r in attempts if r['quality_valid_v2']), None)
        if chosen is not None:
            iid, condition, _ = key
            slot_valid[iid, condition].append(chosen)
    instances = sorted({r['instance_id'] for r in rescored})
    selected = []
    for iid in instances:
        template = next((r for r in rescored if r['instance_id'] == iid))
        for condition in CONDITIONS:
            valid = sorted(slot_valid[iid, condition], key=lambda r: int(r['slot']))
            if not valid:
                selected.append({'instance_id': iid, 'manifest_index': template['manifest_index'], 'generator_key': template['generator_key'], 'sample_id': template['sample_id'], 'domain': template['domain'], 'condition': condition, 'status': 'ABSTAIN', 'quality_valid_slots': 0, 'output_text': template['source_text'], 'selected_slot': None, 'selected_attempt': None, 'semantic_similarity': 1.0, 'edit_fraction': 0.0, 'length_ratio': 1.0, 'mean_control_tv': 0.0})
                continue
            if condition.startswith('vanilla_'):
                chosen = valid[0]
                selection_rule = 'first_quality_valid_slot'
            else:
                chosen = min(valid, key=lambda r: (float(r['trajectory_typicality']), int(r['slot'])))
                selection_rule = 'minimum_condition_specific_trajectory_typicality'
            selected.append({'instance_id': iid, 'manifest_index': chosen['manifest_index'], 'generator_key': chosen['generator_key'], 'sample_id': chosen['sample_id'], 'domain': chosen['domain'], 'condition': condition, 'status': 'REWRITE', 'selection_rule': selection_rule, 'quality_valid_slots': len(valid), 'output_text': chosen['text'], 'selected_slot': int(chosen['slot']), 'selected_attempt': int(chosen['attempt']), 'semantic_similarity': float(chosen['semantic_similarity']), 'edit_fraction': float(chosen['edit_fraction']), 'length_ratio': float(chosen['length_ratio']), 'trajectory_typicality': float(chosen['trajectory_typicality']), 'mean_control_tv': float(chosen['mean_control_tv'])})
    if len(selected) != 285:
        raise RuntimeError(f'Expected 285 selections, got {len(selected)}')
    write_jsonl(SELECTED_OUT, selected)
    print()
    print('=' * 100)
    print('CORRECTED DOCUMENT COVERAGE')
    print('=' * 100)
    coverage_summary = {}
    for condition in CONDITIONS:
        subset = [r for r in selected if r['condition'] == condition]
        rewrites = [r for r in subset if r['status'] == 'REWRITE']
        ratios = [r['length_ratio'] for r in rewrites]
        by_domain = Counter((r['domain'] for r in rewrites))
        print()
        print(condition)
        print(f'  rewrite coverage: {len(rewrites)}/24 ({len(rewrites) / 95:.1%})')
        print('  domains:', dict(by_domain))
        if ratios:
            print(f'  length ratio mean: {np.mean(ratios):.4f}')
            print(f'  length ratio median: {np.median(ratios):.4f}')
            print(f'  length ratio min: {np.min(ratios):.4f}')
            print(f'  selected <0.80: {sum((x < 0.8 for x in ratios))}')
            print(f'  selected <0.70: {sum((x < 0.7 for x in ratios))}')
        coverage_summary[condition] = {'rewrites': len(rewrites), 'coverage': len(rewrites) / 95, 'domains': dict(by_domain), 'length_ratio_mean': float(np.mean(ratios)) if ratios else None, 'length_ratio_median': float(np.median(ratios)) if ratios else None, 'length_ratio_min': float(np.min(ratios)) if ratios else None, 'selected_below_0_80': int(sum((x < 0.8 for x in ratios))), 'selected_below_0_70': int(sum((x < 0.7 for x in ratios)))}
    selection_map = {(r['instance_id'], r['condition']): r for r in selected}
    matched_n0_n1 = [iid for iid in instances if selection_map[iid, 'hpcd_n0']['status'] == 'REWRITE' and selection_map[iid, 'vanilla_t11']['status'] == 'REWRITE']
    matched_all = [iid for iid in instances if all((selection_map[iid, condition]['status'] == 'REWRITE' for condition in CONDITIONS))]
    print()
    print('=' * 100)
    print('MATCHED COVERAGE AFTER VALIDATOR FIX')
    print('=' * 100)
    print('n0+n1:', f'{len(matched_n0_n1)}/24', f'({len(matched_n0_n1) / 95:.1%})')
    print('Vanilla+n0+n1:', f'{len(matched_all)}/24', f'({len(matched_all) / 95:.1%})')
    summary = {'status': 'COMPLETE', 'input_sha256': actual_sha, 'change': 'Source entities are identified once with spaCy; preservation is checked by normalized exact surface occurrence in raw output rather than requiring spaCy to independently re-recognize the entity in output.', 'semantic_threshold_unchanged': 0.92, 'coverage': coverage_summary, 'matched_n0_n1': matched_n0_n1, 'matched_all_three': matched_all, 'rescored_candidates_sha256': sha256_file(SCORED_OUT), 'selected_outputs_sha256': sha256_file(SELECTED_OUT)}
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print()
    print('=' * 100)
    print('ARTIFACT HASHES')
    print('=' * 100)
    print('rescored:', sha256_file(SCORED_OUT))
    print('selected:', sha256_file(SELECTED_OUT))
    print('summary:', sha256_file(SUMMARY_OUT))
if __name__ == '__main__':
    main()

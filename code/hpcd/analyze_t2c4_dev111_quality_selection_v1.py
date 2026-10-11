from __future__ import annotations
import os

import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
INPUT = ROOT / 'results/hpcd/t2c4_dev111_threegen_candidates_v3/candidates.jsonl'
EXPECTED_INPUT_SHA = '9027ef59f6c1cb3dc70e1f6155871ba46b3fdedf9dd4df0ddaa46d101763eb07'
OUT_DIR = ROOT / 'results/hpcd/t2c4_dev111_quality_selection_v1'
SCORED_JSONL = OUT_DIR / 'scored_candidates.jsonl'
SELECTED_JSONL = OUT_DIR / 'selected_outputs.jsonl'
SUMMARY_JSON = OUT_DIR / 'summary.json'
MPNET_PATH = Path.home() / '.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130'
SEMANTIC_THRESHOLD = 0.92
CONDITIONS = ('vanilla', 'hpcd_n0', 'hpcd_n1')
K = 4
ATTEMPTS = 2

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_jsonl(path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def norm(text):
    return re.sub('\\s+', ' ', text.strip().lower())

def extract_numbers(text):
    return sorted(re.findall('(?<!\\w)[+-]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?%?(?!\\w)', text))

def extract_dates(text):
    values = []
    patterns = ['\\b(?:19|20)\\d{2}\\b', '\\b\\d{1,2}[/-]\\d{1,2}[/-](?:\\d{2}|\\d{4})\\b', '\\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\\s+\\d{1,2}(?:st|nd|rd|th)?(?:,\\s*(?:19|20)\\d{2})?\\b']
    for pattern in patterns:
        values.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return sorted((norm(x) for x in values))

def extract_entities(nlp, text):
    ignore = {'CARDINAL', 'ORDINAL', 'PERCENT', 'MONEY', 'DATE', 'TIME', 'QUANTITY'}
    return sorted((norm(ent.text) for ent in nlp(text).ents if ent.label_ not in ignore and ent.text.strip()))

def preserves(source_items, output_items):
    remaining = list(output_items)
    for item in source_items:
        if item not in remaining:
            return False
        remaining.remove(item)
    return True

def lexical_edit_fraction(source, output):
    return float(1.0 - SequenceMatcher(None, norm(source), norm(output)).ratio())

class MPNetEncoder:

    def __init__(self):
        cfg_path = MPNET_PATH / 'sentence_bert_config.json'
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            self.max_length = int(cfg.get('max_seq_length', 384))
        else:
            self.max_length = 384
        self.tokenizer = AutoTokenizer.from_pretrained(MPNET_PATH, local_files_only=True)
        self.model = AutoModel.from_pretrained(MPNET_PATH, local_files_only=True)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        print('MPNet device:', self.device)

    def encode(self, texts, batch_size=16):
        all_vecs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                out = self.model(**encoded)
                mask = encoded['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).to(out.last_hidden_state.dtype)
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-09)
                pooled = F.normalize(pooled, p=2, dim=1)
            all_vecs.append(pooled.detach().cpu().numpy())
            print(f'MPNet {min(start + batch_size, len(texts))}/{len(texts)}', flush=True)
        return np.concatenate(all_vecs, axis=0)

def mean_or_none(values):
    if not values:
        return None
    return float(np.mean(values))

def median_or_none(values):
    if not values:
        return None
    return float(np.median(values))

def main():
    print('=' * 100)
    print('T2C.3 — FULL 7992-CANDIDATE QUALITY + SELECTION')
    print('=' * 100)
    input_sha = sha256_file(INPUT)
    print('input SHA256:', input_sha)
    if input_sha != EXPECTED_INPUT_SHA:
        raise RuntimeError('Frozen candidate pool SHA mismatch')
    if OUT_DIR.exists():
        existing = list(OUT_DIR.iterdir())
        if existing:
            raise RuntimeError(f'Refusing overwrite: {OUT_DIR}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(INPUT)
    if len(rows) != 7992:
        raise RuntimeError(f'Expected 7992 rows; got {len(rows)}')
    condition_counts = Counter((r['condition'] for r in rows))
    print('condition counts:', dict(condition_counts))
    for condition in CONDITIONS:
        if condition_counts[condition] != 2664:
            raise RuntimeError(f'{condition}: expected 192 rows')
    instances = sorted({r['instance_id'] for r in rows})
    if len(instances) != 333:
        raise RuntimeError(f'Expected 24 docs; got {len(instances)}')
    keys = Counter(((r['instance_id'], r['condition'], int(r['slot']), int(r['attempt'])) for r in rows))
    expected_keys = {(instance_id, condition, slot, attempt) for instance_id in instances for condition in CONDITIONS for slot in range(K) for attempt in range(ATTEMPTS)}
    if set(keys) != expected_keys:
        raise RuntimeError('Candidate key grid is incomplete')
    if any((count != 1 for count in keys.values())):
        raise RuntimeError('Duplicate candidate key detected')
    source_by_instance = {}
    for row in rows:
        iid = row['instance_id']
        source = row['source_text']
        if iid in source_by_instance and source_by_instance[iid] != source:
            raise RuntimeError(f'Source inconsistency: {iid}')
        source_by_instance[iid] = source
    nlp = spacy.load('en_core_web_sm')
    encoder = MPNetEncoder()
    source_order = instances
    source_texts = [source_by_instance[iid] for iid in source_order]
    candidate_texts = [r['text'] for r in rows]
    print()
    print('Encoding', len(source_texts), 'sources +', len(candidate_texts), 'candidates')
    embeddings = encoder.encode(source_texts + candidate_texts, batch_size=16)
    source_embeddings = {iid: embeddings[i] for i, iid in enumerate(source_order)}
    candidate_embeddings = embeddings[len(source_order):]
    source_fact_cache = {}
    for iid in source_order:
        source = source_by_instance[iid]
        source_fact_cache[iid] = {'numbers': extract_numbers(source), 'dates': extract_dates(source), 'entities': extract_entities(nlp, source)}
    scored = []
    for i, row in enumerate(rows):
        rec = dict(row)
        iid = row['instance_id']
        source = row['source_text']
        text = row['text']
        similarity = float(np.dot(source_embeddings[iid], candidate_embeddings[i]))
        facts = source_fact_cache[iid]
        number_ok = preserves(facts['numbers'], extract_numbers(text))
        date_ok = preserves(facts['dates'], extract_dates(text))
        entity_ok = preserves(facts['entities'], extract_entities(nlp, text))
        semantic_ok = bool(similarity >= SEMANTIC_THRESHOLD)
        quality_valid = bool(text.strip() and semantic_ok and number_ok and date_ok and entity_ok)
        rec.update({'semantic_similarity': similarity, 'semantic_ok': semantic_ok, 'number_ok': bool(number_ok), 'date_ok': bool(date_ok), 'entity_ok': bool(entity_ok), 'quality_valid': quality_valid, 'edit_fraction': lexical_edit_fraction(source, text), 'source_char_length': len(source), 'output_char_length': len(text), 'length_ratio': len(text) / max(1, len(source))})
        scored.append(rec)
    write_jsonl(SCORED_JSONL, scored)
    grouped = defaultdict(list)
    for row in scored:
        grouped[row['instance_id'], row['condition'], int(row['slot'])].append(row)
    valid_slot_candidates = defaultdict(list)
    slot_outcomes = []
    for iid in instances:
        for condition in CONDITIONS:
            for slot in range(K):
                attempts = sorted(grouped[iid, condition, slot], key=lambda r: int(r['attempt']))
                if len(attempts) != ATTEMPTS:
                    raise RuntimeError(f'Bad attempts for {iid}/{condition}/{slot}')
                chosen = next((r for r in attempts if r['quality_valid']), None)
                outcome = {'instance_id': iid, 'condition': condition, 'slot': slot, 'valid': chosen is not None, 'chosen_attempt': int(chosen['attempt']) if chosen else None}
                slot_outcomes.append(outcome)
                if chosen is not None:
                    valid_slot_candidates[iid, condition].append(chosen)
    selected = []
    by_instance_condition = {}
    for iid in instances:
        metadata = next((r for r in scored if r['instance_id'] == iid))
        source = metadata['source_text']
        for condition in CONDITIONS:
            valid = sorted(valid_slot_candidates[iid, condition], key=lambda r: int(r['slot']))
            if not valid:
                rec = {'instance_id': iid, 'manifest_index': metadata['manifest_index'], 'generator_key': metadata['generator_key'], 'sample_id': metadata['sample_id'], 'domain': metadata['domain'], 'condition': condition, 'status': 'ABSTAIN', 'reason': 'no_quality_valid_candidate', 'quality_valid_slots': 0, 'output_text': source, 'selected_slot': None, 'selected_attempt': None, 'selected_seed': None, 'semantic_similarity': 1.0, 'edit_fraction': 0.0, 'trajectory_typicality': None, 'mean_control_tv': 0.0}
            else:
                if condition == 'vanilla':
                    chosen = valid[0]
                    selection_rule = 'first_quality_valid_slot'
                else:
                    chosen = min(valid, key=lambda r: (float(r['trajectory_typicality']), int(r['slot'])))
                    selection_rule = 'minimum_condition_specific_trajectory_typicality'
                rec = {'instance_id': iid, 'manifest_index': chosen['manifest_index'], 'generator_key': chosen['generator_key'], 'sample_id': chosen['sample_id'], 'domain': chosen['domain'], 'condition': condition, 'status': 'REWRITE', 'reason': 'quality_valid_candidate_available', 'selection_rule': selection_rule, 'quality_valid_slots': len(valid), 'output_text': chosen['text'], 'selected_slot': int(chosen['slot']), 'selected_attempt': int(chosen['attempt']), 'selected_seed': int(chosen['seed']), 'semantic_similarity': float(chosen['semantic_similarity']), 'edit_fraction': float(chosen['edit_fraction']), 'length_ratio': float(chosen['length_ratio']), 'trajectory_typicality': float(chosen['trajectory_typicality']), 'mean_control_tv': float(chosen['mean_control_tv']), 'mean_support_size': float(chosen['mean_support_size']), 'controllable_fraction': float(chosen['controllable_fraction'])}
            selected.append(rec)
            by_instance_condition[iid, condition] = rec
    if len(selected) != 999:
        raise RuntimeError(f'Expected 72 final rows; got {len(selected)}')
    write_jsonl(SELECTED_JSONL, selected)
    summary_conditions = {}
    print()
    print('=' * 100)
    print('FINAL COVERAGE')
    print('=' * 100)
    for condition in CONDITIONS:
        subset = [r for r in selected if r['condition'] == condition]
        rewritten = [r for r in subset if r['status'] == 'REWRITE']
        abstained = [r for r in subset if r['status'] == 'ABSTAIN']
        sims = [r['semantic_similarity'] for r in rewritten]
        edits = [r['edit_fraction'] for r in rewritten]
        tvs = [r['mean_control_tv'] for r in rewritten]
        typs = [r['trajectory_typicality'] for r in rewritten if r['trajectory_typicality'] is not None]
        domain_rewrites = Counter((r['domain'] for r in rewritten))
        summary_conditions[condition] = {'rewrites': len(rewritten), 'abstentions': len(abstained), 'coverage': len(rewritten) / 24.0, 'mean_semantic_similarity': mean_or_none(sims), 'median_semantic_similarity': median_or_none(sims), 'mean_edit_fraction': mean_or_none(edits), 'median_edit_fraction': median_or_none(edits), 'mean_control_tv': mean_or_none(tvs), 'mean_selected_condition_typicality': mean_or_none(typs), 'domain_rewrite_counts': dict(domain_rewrites)}
        print()
        print(condition)
        print(f'  rewrite coverage: {len(rewritten)}/24 ({len(rewritten) / 24:.1%})')
        print(f'  abstentions: {len(abstained)}/24')
        print(f'  mean MPNet: {mean_or_none(sims)}')
        print(f'  mean edit fraction: {mean_or_none(edits)}')
        print(f'  mean TV: {mean_or_none(tvs)}')
        print('  domain rewrites:', dict(domain_rewrites))
    matched_all = []
    matched_n0_n1 = []
    for iid in instances:
        v = by_instance_condition[iid, 'vanilla']
        n0 = by_instance_condition[iid, 'hpcd_n0']
        n1 = by_instance_condition[iid, 'hpcd_n1']
        if v['status'] == 'REWRITE' and n0['status'] == 'REWRITE' and (n1['status'] == 'REWRITE'):
            matched_all.append(iid)
        if n0['status'] == 'REWRITE' and n1['status'] == 'REWRITE':
            matched_n0_n1.append(iid)
    print()
    print('=' * 100)
    print('MATCHED COVERAGE')
    print('=' * 100)
    print('Vanilla + n0 + n1:', f'{len(matched_all)}/24', f'({len(matched_all) / 24:.1%})')
    print('n0 + n1:', f'{len(matched_n0_n1)}/24', f'({len(matched_n0_n1) / 24:.1%})')
    raw_validity = {}
    for condition in CONDITIONS:
        subset = [r for r in scored if r['condition'] == condition]
        valid = [r for r in subset if r['quality_valid']]
        raw_validity[condition] = {'valid_candidates': len(valid), 'total_candidates': len(subset), 'candidate_valid_rate': len(valid) / len(subset), 'semantic_pass_rate': sum((r['semantic_ok'] for r in subset)) / len(subset), 'number_pass_rate': sum((r['number_ok'] for r in subset)) / len(subset), 'date_pass_rate': sum((r['date_ok'] for r in subset)) / len(subset), 'entity_pass_rate': sum((r['entity_ok'] for r in subset)) / len(subset)}
    summary = {'status': 'COMPLETE', 'stage': 'T2C.3 full quality-first selection', 'input_sha256': input_sha, 'semantic_threshold': SEMANTIC_THRESHOLD, 'K': K, 'attempts': ATTEMPTS, 'quality_gate': {'mpnet_similarity_ge': SEMANTIC_THRESHOLD, 'numbers_preserved': True, 'dates_preserved': True, 'named_entities_preserved': True, 'nonempty': True}, 'selection_policy': {'within_slot': 'first quality-valid attempt', 'vanilla': 'first quality-valid slot; no HJTM selection', 'hpcd_n0': 'minimum n0 trajectory typicality among valid slots', 'hpcd_n1': 'minimum n1 trajectory typicality among valid slots', 'abstention': 'return source unchanged if no valid slot'}, 'raw_candidate_validity': raw_validity, 'conditions': summary_conditions, 'matched_all_three': len(matched_all), 'matched_n0_n1': len(matched_n0_n1), 'matched_all_three_ids': matched_all, 'matched_n0_n1_ids': matched_n0_n1, 'scored_candidates_sha256': sha256_file(SCORED_JSONL), 'selected_outputs_sha256': sha256_file(SELECTED_JSONL)}
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print()
    print('=' * 100)
    print('ARTIFACT HASHES')
    print('=' * 100)
    print('scored candidates:', sha256_file(SCORED_JSONL))
    print('selected outputs:', sha256_file(SELECTED_JSONL))
    print('summary:', sha256_file(SUMMARY_JSON))
    print()
    print('=' * 100)
    print('T2C.3 QUALITY + SELECTION COMPLETE')
    print('=' * 100)
if __name__ == '__main__':
    main()

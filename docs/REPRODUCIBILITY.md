# Reproducibility checklist

## Frozen experiment summary

- Reference LM and paraphraser: Qwen2.5-3B-Instruct (distinct conditioning during decoding).
- Token mass bins: 6. Entropy regions: 3. Final HJTM order: n=0.
- Decoder: projected squared standardized human NLL deviation; beta=0.22; T=1.0; top-p=0.95.
- Candidate budget: K=4, two attempts per slot, deterministic SHA-256-derived seeds; first token uncontrolled.
- Quality gate: MPNet cosine >=0.92; Source numbers, dates, and source-side normalized named entities must be preserved; length not a validity gate; fallback to unchanged Source.
- Selection: HPCD picks minimum T0 among quality-valid candidates, Matched takes first valid.
- Data: Primary/Fresh95 and Replication/Confirm48 each include 48 base documents and 95 machine-source realizations.
- Statistical inference: domain-stratified RAID-base cluster bootstrap, 50,000 replicates, seed 2027.

## What is available

Frozen core code, held-out drivers, a reference occupancy artifact, configuration records, evaluation helpers, historical baseline integrations and selected aggregate tables.

## What remains external

The original RAID inputs, the actual frozen held-out input manifests including their source texts and checksums, pretrained model weights, frozen external detector checkpoints, and some large/intermediate candidate arrays are not part of this ZIP. Their absence prevents a one-command reproduction of the full paper. Private human-annotation forms are also omitted. No fresh generation has been conducted by packaging this repository.

## Provenance

Only machine-specific paths and one configuration path field were changed in the release snapshot. See `docs/PROVENANCE.md` for original hashes. Some of the frozen scripts require local environment paths and additional intermediate artifacts; source integrity and syntax validation alone cannot guarantee end-to-end result reproduction.

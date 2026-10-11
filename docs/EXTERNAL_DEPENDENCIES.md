# External methods and models

The repository intentionally contains **only locally authored integrations**, not copies of external baseline implementations. Dependencies were evaluated under their separate release terms.

| Component | Local integration | External prerequisite |
|---|---|---|
| CoPA | `code/copa/run_copa3b_fresh95_full_v1.py` | Authors' CoPA source and matched Qwen2.5-3B-Instruct checkpoint |
| DIPPER | `dipper_friend_toolkit_v1/dipper_fresh95_confirm48_multigpu_v1.py` and `code/extension_oct2026_v1/dipper/` | Official DIPPER checkpoint and T5-v1_1-xxl tokenizer |
| Fast-DetectGPT | `code/evaluate_fastdetectgpt_neo27_mac_v1.py` | Original detector implementation and its scoring/reference models |
| RADAR | `code/evaluate_radar_detector_v1.py` | Frozen RADAR model/checkpoint |
| SurpMark | `results/detectors/fresh95_final_v1/surpmark/final_v1/run_surpmark_full_v1.py` | Original SurpMark implementation and model |
| MAGE | `code/extension_oct2026_v1/run_mage_frozen523_v1.py` | Frozen Longformer checkpoint/tokenizer |

Baseline implementations were **not bundled or modified**. Check the original papers, official repositories, exact checkpoints and software licenses before re-running. Scripts may assume different compute environments (macOS MPS, Linux CUDA) and fixed, historically-local directories. Their porting has not been validated in this snapshot.

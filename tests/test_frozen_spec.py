import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class FrozenSpecTests(unittest.TestCase):
    def test_reference_model_and_state_bins(self):
        model = json.loads((ROOT / "results/hjtm/final_hjtm_v1/hjtm_final_v1.json").read_text())
        self.assertEqual(model["state_definition"]["B"], 6)
        self.assertEqual(model["state_definition"]["E"], 3)
        self.assertEqual(model["model_name"], "HJTM_FINAL_V1")

    def test_freeze_parameters(self):
        freeze = json.loads((ROOT / "configs/hpcd/FINAL_METHOD_FREEZE.json").read_text())
        self.assertEqual(freeze["decoding"]["K"], 4)
        self.assertEqual(freeze["decoding"]["attempts_per_slot"], 2)
        self.assertEqual(freeze["decoding"]["temperature"], 1.0)
        self.assertEqual(freeze["decoding"]["top_p"], 0.95)
        self.assertEqual(freeze["method"]["controller_order"], 0)
        self.assertEqual(freeze["method"]["beta"], 0.22)

if __name__ == "__main__":
    unittest.main()

"""Synthetic-vocabulary smoke test: no model checkpoint or dataset required."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code' / 'hpcd'))

try:
    import torch
    from hpcd_controller_v1 import HJTMModel
    from hpcd_reference_aligned_v2 import ReferenceAlignedHPCDController
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'torch and numpy are needed')
class FrozenDecoderSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hjtm = HJTMModel(ROOT / 'results/hjtm/final_hjtm_v1/hjtm_final_v1.json')
        cls.controller = ReferenceAlignedHPCDController(cls.hjtm)
        cls.base = torch.tensor([3.0, 2.5, 2., 1.6, 1.2, .8, .4, 0, -.4, -.8, -1, -1.5])
        cls.ref = torch.tensor([1.0, 3.0, 2., .4, .5, -.3, 1.2, -1., .2, 2.5, -.4, .3])

    def call_controller(self, beta):
        return self.controller.control_reference_aligned(
            base_logits=self.base,
            reference_logits=self.ref,
            history=[],
            cumulative_cost=0.0,
            order=0,
            beta=beta,
            domain='news',
            mode='typicality',
            terminal_token_ids=[1],
            temperature=1.0,
            top_p=.95,
        )

    def test_normalization_and_terminal_mass_preservation(self):
        result = self.call_controller(.22)
        self.assertAlmostEqual(float(result['distribution'].sum()), 1.0, places=5)
        self.assertAlmostEqual(
            float(result['terminal_probability_base']),
            float(result['terminal_probability_controlled']), places=5)
        self.assertGreater(result['support_size'], 0)

    def test_matched_baseline_is_unchanged_at_zero_bias(self):
        result = self.call_controller(0.0)
        self.assertTrue(torch.allclose(result['distribution'], result['base_distribution'], atol=1e-6))

    def test_human_model_is_normalized(self):
        probs = self.hjtm.conditional_vector(history=[], entropy_state=0, order=0)
        self.assertEqual(len(probs), 6)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=8)


if __name__ == '__main__':
    unittest.main()

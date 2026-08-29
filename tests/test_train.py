from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train as tm

ONNX = ROOT / "model_assets/BT4-1024x15x32h-swa-6147500_with_embedding.onnx"  # from scripts/prepare_network.py


def _inputs(seed=0, rows=4, moves=6):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, moves, generator=g)
    mask = torch.ones(rows, moves, dtype=torch.bool)
    mask[0, -1] = False
    masked = torch.where(mask, logits, torch.full_like(logits, -1e9))
    wdl = torch.softmax(torch.randn(rows, 3, generator=g), dim=-1)
    target_idx = torch.tensor([0, 1, 2, 3])
    rho = torch.tensor([1.0, 0.5, 0.25, 1.0])
    adv = torch.tensor([0.1, -0.2, 0.3, 0.0])
    wdl_target = torch.softmax(torch.randn(rows, 3, generator=g), dim=-1)
    base_policy = torch.softmax(masked + 0.3 * torch.randn(rows, moves, generator=g), dim=-1)
    base_wdl = torch.softmax(torch.randn(rows, 3, generator=g), dim=-1)
    return masked, mask, wdl, target_idx, rho, adv, wdl_target, base_policy, base_wdl


class StepLossTest(unittest.TestCase):
    def test_terms_match_closed_forms(self) -> None:
        masked, mask, wdl, idx, rho, adv, wdl_t, base_p, base_w = _inputs()
        args = tm.parse_args(["--regularizer", "forward_kl", "--beta", "0.01"])
        loss, parts = tm.step_loss(masked, mask, wdl, idx, rho, adv, wdl_t, base_p, base_w, args)

        log_p = torch.log_softmax(masked, dim=-1)
        policy = -(rho * adv * log_p[torch.arange(4), idx]).mean()
        wdl_ce = (rho * (-(wdl_t * torch.log(wdl)).sum(-1) + (wdl_t * torch.log(wdl_t)).sum(-1))).mean()
        fwd = (torch.special.xlogy(base_p, base_p) - base_p * log_p).sum(-1).mean()
        base_wdl = (base_w * (torch.log(base_w) - torch.log(wdl))).sum(-1).mean()

        self.assertTrue(torch.allclose(parts["policy"], policy, atol=1e-6))
        self.assertTrue(torch.allclose(parts["wdl"], wdl_ce, atol=1e-6))
        self.assertTrue(torch.allclose(parts["reg"], fwd, atol=1e-6))
        self.assertTrue(torch.allclose(parts["base_wdl"], base_wdl, atol=1e-6))
        expected = policy + 0.01 * fwd + 0.2 * wdl_ce + 0.1 * 0.2 * base_wdl
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_reverse_and_entropy_regularizers(self) -> None:
        masked, mask, wdl, idx, rho, adv, wdl_t, base_p, base_w = _inputs(seed=1)
        log_p = torch.log_softmax(masked, dim=-1)
        p = log_p.exp()

        rev = tm.step_loss(masked, mask, wdl, idx, rho, adv, wdl_t, base_p, base_w,
                           tm.parse_args(["--regularizer", "reverse_kl"]))[1]
        p8, b8 = p.clamp_min(1e-8), base_p.clamp_min(1e-8)
        self.assertTrue(torch.allclose(rev["reg"], (p8 * (torch.log(p8) - torch.log(b8))).sum(-1).mean(), atol=1e-5))
        self.assertTrue(torch.allclose(rev["base_wdl"], (wdl * (torch.log(wdl) - torch.log(base_w))).sum(-1).mean(), atol=1e-5))

        ent = tm.step_loss(masked, mask, wdl, idx, rho, adv, wdl_t, base_p, base_w,
                           tm.parse_args(["--regularizer", "entropy"]))[1]
        n_legal = mask.float().sum(-1)
        kl_uniform = ((p * torch.log(p.clamp_min(1e-8))).sum(-1) + torch.log(n_legal)).mean()
        self.assertTrue(torch.allclose(ent["reg"], kl_uniform, atol=1e-5))

        none = tm.step_loss(masked, mask, wdl, idx, rho, adv, wdl_t, base_p, base_w,
                            tm.parse_args(["--regularizer", "none", "--base-wdl-kl-weight", "0"]))[1]
        self.assertEqual(float(none["reg"]), 0.0)
        self.assertEqual(float(none["base_wdl"]), 0.0)

    def test_defaults(self) -> None:
        a = tm.parse_args([])
        self.assertEqual((a.regularizer, a.beta, a.wdl_kl_weight, a.base_wdl_kl_weight, a.ema_decay), ("forward_kl", 0.01, 0.2, 0.1, 0.995))
        self.assertEqual((a.lr, a.lc0_last_block_lr, a.policy_head_lr, a.wdl_head_lr), (3e-6, 1e-5, 1e-5, 3e-5))
        self.assertEqual((a.num_games, a.total_gradient_steps, a.warmup_steps, a.max_plies), (1024, 2000, 200, 512))
        self.assertEqual(a.sampling_temperature_wdl_source, "online")


class MinimalTrainerSmokeTest(unittest.TestCase):
    def test_one_step_on_cpu(self) -> None:
        if not ONNX.exists():
            raise unittest.SkipTest("network asset not prepared (scripts/prepare_network.py)")
        with tempfile.TemporaryDirectory() as tmp:
            tm.main([
                "--onnx", str(ONNX), "--device", "cpu", "--num-games", "2", "--minibatch-size", "2",
                "--forward-batch-size", "2", "--total-gradient-steps", "1", "--warmstart-max-full-moves", "1",
                "--sample-with-wdl-entropy-temperature", "--save-every", "1", "--output-dir", tmp,
            ])
            payload = torch.load(Path(tmp) / "model.pt", map_location="cpu", weights_only=False)
            metrics = [json.loads(line) for line in (Path(tmp) / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(payload["step"], 1)
        self.assertIn("ema_model_state_dict", payload)
        self.assertIn("model_state_dict", payload)
        self.assertEqual(metrics[0]["step"], 1)
        self.assertTrue(math.isfinite(metrics[0]["loss"]))
        self.assertEqual(metrics[0]["positions"], 2)


if __name__ == "__main__":
    unittest.main()

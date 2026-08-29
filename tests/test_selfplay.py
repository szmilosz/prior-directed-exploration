from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import chess
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import selfplay as sp


def _middlegame() -> chess.Board:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        board.push_san(san)
    return board


class BatchingTest(unittest.TestCase):
    def test_batch_shapes_and_legal_mask(self) -> None:
        boards = [chess.Board(), _middlegame()]
        x, mask, meta, moves, idx = sp.build_batch(boards, "cpu")
        self.assertEqual(tuple(mask.shape), (2, sp.NUM_MOVE_SLOTS))
        self.assertEqual(x.shape[0], 2)
        for row, board in enumerate(boards):
            self.assertEqual(len(moves[row]), board.legal_moves.count())
            self.assertEqual(int(mask[row].sum()), len(set(idx[row])))
            self.assertTrue(all(mask[row, i] for i in idx[row]))
        self.assertEqual(tuple(meta["history_piece_planes"].shape), (2, 8, 12, 8, 8))


class SamplingTest(unittest.TestCase):
    def test_promotion_offsets_apply_per_piece(self) -> None:
        board = chess.Board("8/1P4k1/8/8/8/8/6K1/8 w - - 0 1")  # b7-b8 promotes
        _, mask, _, moves, idx = sp.build_batch([board], "cpu")
        logits = torch.zeros(1, sp.NUM_MOVE_SLOTS)
        offsets = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        scores = sp.legal_move_scores(logits[0], moves[0], idx[0], offsets[0])
        for move, score in zip(moves[0], scores):
            row = sp.PROMOTION_OFFSET_ROW.get(move.promotion)
            expected = 0.0 if row is None else float(offsets[0, row, chess.square_file(move.to_square)])
            self.assertEqual(float(score), expected)
        self.assertEqual(sp.argmax_move(logits[0], moves[0], idx[0], offsets[0]).promotion, chess.BISHOP)
        applied = sp.apply_promotion_offsets(logits, moves, idx, {"promotion_offsets": offsets})
        self.assertEqual(tuple(applied.shape), tuple(logits.shape))
        illegal = ~mask[0]
        self.assertTrue(torch.equal(sp.masked_logits(logits, mask, moves, idx, None)[0][illegal],
                                    torch.full((int(illegal.sum()),), -1e9)))

    def test_sample_move_probabilities_and_temperature(self) -> None:
        board = _middlegame()
        _, _, _, moves, idx = sp.build_batch([board], "cpu")
        logits = torch.randn(sp.NUM_MOVE_SLOTS, generator=torch.Generator().manual_seed(3))
        scores = sp.legal_move_scores(logits, moves[0], idx[0])
        torch.manual_seed(7)
        move, p_model, p_proposal, local = sp.sample_move(logits, moves[0], idx[0])
        self.assertIs(move, moves[0][local])
        self.assertEqual(p_model, p_proposal)
        self.assertAlmostEqual(p_model, float(torch.softmax(scores, dim=0)[local]), places=6)
        torch.manual_seed(7)
        _, p_model_t, p_proposal_t, local_t = sp.sample_move(logits, moves[0], idx[0], temperature=0.1)
        self.assertAlmostEqual(p_proposal_t, float(torch.softmax(scores / 0.1, dim=0)[local_t]), places=6)
        self.assertAlmostEqual(p_model_t, float(torch.softmax(scores, dim=0)[local_t]), places=6)

    def test_importance_weight(self) -> None:
        self.assertAlmostEqual(sp.importance_weight(0.1, 0.4), 0.25, places=9)
        self.assertEqual(sp.importance_weight(0.5, 0.25), 1.0)
        self.assertEqual(sp.importance_weight(0.3, 0.3), 1.0)
        self.assertAlmostEqual(sp.importance_weight(0.5, 0.25, clip_max=5.0), 2.0, places=9)
        self.assertEqual(sp.importance_weight(0.5, 0.25, clip_max=0.0), 0.0)

    def test_wdl_entropy(self) -> None:
        self.assertAlmostEqual(sp.wdl_entropy((1 / 3, 1 / 3, 1 / 3)), math.log(3), places=6)
        self.assertEqual(sp.wdl_entropy((1.0, 0.0, 0.0)), 0.0)
        self.assertGreater(sp.wdl_entropy((0.9, 0.05, 0.05)), 0.0)

    def test_kl_to_uniform(self) -> None:
        probs = torch.tensor([[0.75, 0.25, 0.0]])
        mask = torch.tensor([[True, True, False]])
        expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
        self.assertAlmostEqual(float(sp.kl_to_uniform_over_legal_moves(probs, mask)), expected, places=6)
        self.assertAlmostEqual(float(sp.kl_to_uniform_over_legal_moves(torch.tensor([[0.5, 0.5, 0.0]]), mask)), 0.0, places=6)


class ResultsTest(unittest.TestCase):
    def test_game_results(self) -> None:
        mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")  # fool's mate
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        fifty = chess.Board("8/8/8/4k3/8/8/8/4K2R w K - 100 80")
        self.assertEqual(sp.game_result(chess.Board()), "*")
        self.assertEqual(sp.game_result(mate), "0-1")
        self.assertEqual(sp.game_result(stalemate), "1/2-1/2")
        self.assertEqual(sp.game_result(fifty), "1/2-1/2")
        self.assertTrue(sp.is_terminal(mate) and sp.is_terminal(fifty) and not sp.is_terminal(chess.Board()))
        self.assertTrue(torch.equal(sp.wdl_from_result(mate, "0-1"), torch.tensor([0.0, 0.0, 1.0])))  # White to move, lost
        self.assertTrue(torch.equal(sp.wdl_from_result(chess.Board(), "1-0"), torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(sp.wdl_from_result(stalemate, "1/2-1/2"), torch.tensor([0.0, 1.0, 0.0])))

    def test_pov_flip(self) -> None:
        wdl = torch.tensor([0.7, 0.2, 0.1])
        white = chess.Board()
        black = chess.Board()
        black.push_san("e4")
        self.assertTrue(torch.equal(sp.wdl_to_pov(wdl, black, white), torch.tensor([0.1, 0.2, 0.7])))
        self.assertTrue(torch.equal(sp.wdl_to_pov(wdl, white, white), wdl))
        self.assertTrue(torch.equal(sp.wdl_to_pov(wdl, white, _middlegame()), wdl))  # both White to move
        self.assertAlmostEqual(sp.expected_score(wdl), 0.8, places=6)

    def test_warmup_lr(self) -> None:
        self.assertEqual(sp.warmup_lr(0, 3e-6, 200), 0.0)
        self.assertAlmostEqual(sp.warmup_lr(100, 3e-6, 200), 1.5e-6)
        self.assertEqual(sp.warmup_lr(200, 3e-6, 200), 3e-6)
        self.assertEqual(sp.warmup_lr(2000, 3e-6, 200), 3e-6)
        self.assertEqual(sp.warmup_lr(5, 3e-6, 0), 3e-6)


if __name__ == "__main__":
    unittest.main()

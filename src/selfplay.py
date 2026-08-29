"""Self-play primitives: batching, move sampling, importance weights, game results,
WDL bookkeeping, model/optimizer construction, checkpoints, and opening warmstart.

Everything here is a pure function of its inputs (no global state), so the training
step in `train.py` reads as the algorithm it implements.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import chess
import numpy as np
import torch

import network as model_module
from chess_encoding import get_input_tensor, get_lc0_meta_from_board, get_move_index

# Row of the model's [3, 8] promotion-offset table for each promotion piece
# (knight promotions use the raw from-to logit).
PROMOTION_OFFSET_ROW = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2}
NUM_MOVE_SLOTS = 4096  # from-square x to-square


@dataclass
class GameState:
    board: chess.Board
    plies: int = 0

    @classmethod
    def new(cls) -> "GameState":
        return cls(board=chess.Board(), plies=0)


# ----------------------------------------------------------------------------- batching


def repetition_flag(board: chess.Board) -> float:
    try:
        return 1.0 if board.is_repetition(2) else 0.0
    except Exception:
        return 0.0


def history_meta(board: chess.Board) -> dict:
    """Lc0 metadata for `board`: castling/side/rule-50 plus the 8-frame history planes
    (missing frames are filled with the oldest available position, as Lc0 does)."""
    perspective_turn = board.turn

    history_boards = []
    temp = board.copy(stack=True)
    history_boards.append(temp.copy(stack=True))
    for _ in range(7):
        if temp.move_stack:
            temp.pop()
            history_boards.append(temp.copy(stack=True))
        else:
            history_boards.append(None)

    oldest_available = None
    for hist_board in reversed(history_boards):
        if hist_board is not None:
            oldest_available = hist_board
            break
    if oldest_available is not None and oldest_available.fen() != chess.STARTING_FEN:
        for idx, hist_board in enumerate(history_boards):
            if hist_board is None:
                history_boards[idx] = oldest_available.copy(stack=True)

    history_piece_planes = np.zeros((8, 12, 8, 8), dtype=np.float32)
    history_repetition = np.zeros((8,), dtype=np.float32)
    for frame_idx, hist_board in enumerate(history_boards):
        if hist_board is None:
            continue
        b = hist_board.copy(stack=False)
        b.turn = perspective_turn
        model_order = get_input_tensor(b).numpy()
        history_piece_planes[frame_idx] = model_order[:, ::-1, :]
        history_repetition[frame_idx] = repetition_flag(hist_board)

    meta = get_lc0_meta_from_board(board, include_repetition=False)
    meta["history_repetition"] = history_repetition
    meta["history_piece_planes"] = history_piece_planes
    return meta


def lc0_meta_batch(meta_items: list[dict], device) -> dict[str, torch.Tensor]:
    stack = lambda key: torch.tensor(np.stack([m[key] for m in meta_items], axis=0), dtype=torch.float32, device=device)
    return {
        "castling": stack("castling"),
        "side_to_move": torch.tensor([m["side_to_move"] for m in meta_items], dtype=torch.float32, device=device),
        "rule50_count": torch.tensor([m["rule50_count"] for m in meta_items], dtype=torch.float32, device=device),
        "history_repetition": stack("history_repetition"),
        "history_piece_planes": stack("history_piece_planes"),
    }


def build_batch(boards: list[chess.Board], device):
    """Returns (input planes, legal mask over the 4096 slots, lc0 meta, legal moves per
    board, legal slot indices per board)."""
    x_list, masks, legal_moves_by_board, legal_indices_by_board, meta_items = [], [], [], [], []
    for board in boards:
        x_list.append(get_input_tensor(board))
        meta_items.append(history_meta(board))
        legal_moves = list(board.legal_moves)
        flip = board.turn == chess.BLACK
        legal_indices = [get_move_index(move.uci(), flip, board=board) for move in legal_moves]
        mask = torch.zeros(NUM_MOVE_SLOTS, dtype=torch.bool)
        if legal_indices:
            mask[torch.tensor(legal_indices, dtype=torch.long)] = True
        masks.append(mask)
        legal_moves_by_board.append(legal_moves)
        legal_indices_by_board.append(legal_indices)
    return (
        torch.stack(x_list).to(device),
        torch.stack(masks).to(device),
        lc0_meta_batch(meta_items, device),
        legal_moves_by_board,
        legal_indices_by_board,
    )


# ------------------------------------------------------------------ move scores and sampling


def promotion_offsets_for_row(aux_outputs, row_idx):
    offsets = aux_outputs.get("promotion_offsets") if isinstance(aux_outputs, dict) else None
    return None if offsets is None else offsets[row_idx]


def legal_move_scores(logits_row, legal_moves, legal_indices, promotion_offsets_row=None):
    """Per-legal-move scores: the from-to logit, plus the promotion offset for Q/R/B promotions."""
    idx_t = torch.tensor(legal_indices, dtype=torch.long, device=logits_row.device)
    legal_logits = logits_row[idx_t]
    if promotion_offsets_row is None:
        return legal_logits
    scores = legal_logits.clone()
    promotion_offsets_row = promotion_offsets_row.to(device=logits_row.device, dtype=logits_row.dtype)
    for move_idx, move in enumerate(legal_moves):
        row = PROMOTION_OFFSET_ROW.get(move.promotion)
        if row is None:
            continue
        scores[move_idx] = scores[move_idx] + promotion_offsets_row[row, chess.square_file(move.to_square)]
    return scores


def apply_promotion_offsets(logits, legal_moves_by_board, legal_indices_by_board, aux_outputs):
    """Add the promotion offsets into the 4096-slot logits (differentiably), for the
    batched softmax used by the losses."""
    offsets = aux_outputs.get("promotion_offsets") if isinstance(aux_outputs, dict) else None
    if offsets is None:
        return logits
    batch_indices, flat_indices, offset_values = [], [], []
    for row_idx in range(logits.shape[0]):
        promo_row = offsets[row_idx]  # [3, 8]
        for move_idx, move in enumerate(legal_moves_by_board[row_idx]):
            row = PROMOTION_OFFSET_ROW.get(move.promotion)
            if row is None:
                continue
            batch_indices.append(row_idx)
            flat_indices.append(legal_indices_by_board[row_idx][move_idx])
            offset_values.append(promo_row[row, chess.square_file(move.to_square)])
    if not offset_values:
        return logits
    delta = torch.zeros_like(logits).index_put(
        (
            torch.tensor(batch_indices, dtype=torch.long, device=logits.device),
            torch.tensor(flat_indices, dtype=torch.long, device=logits.device),
        ),
        torch.stack(offset_values),
        accumulate=True,
    )
    return logits + delta


def masked_logits(logits, mask, legal_moves_by_board, legal_indices_by_board, aux_outputs):
    """4096-slot logits with promotion offsets applied and illegal slots at -1e9."""
    effective = apply_promotion_offsets(logits, legal_moves_by_board, legal_indices_by_board, aux_outputs)
    return torch.where(mask, effective, torch.full_like(effective, -1e9))


def argmax_move(logits_row, legal_moves, legal_indices, promotion_offsets_row=None):
    if not legal_moves:
        return None
    scores = legal_move_scores(logits_row, legal_moves, legal_indices, promotion_offsets_row)
    return legal_moves[int(torch.argmax(scores).item())]


def sample_move(logits_row, legal_moves, legal_indices, promotion_offsets_row=None, temperature=None):
    """Sample a legal move from softmax(scores / temperature) (temperature None = 1).

    Returns (move, model probability, proposal probability, index into legal_moves);
    the two probabilities give the importance ratio pi(a|s) / mu(a|s)."""
    if not legal_moves:
        return None, None, None, None
    scores = legal_move_scores(logits_row, legal_moves, legal_indices, promotion_offsets_row)
    model_probs = torch.softmax(scores, dim=0)
    if temperature is None:
        proposal_probs = model_probs
    else:
        proposal_probs = torch.softmax(scores / max(float(temperature), 1e-8), dim=0)
    sampled = int(torch.multinomial(proposal_probs, num_samples=1).item())
    return legal_moves[sampled], float(model_probs[sampled].item()), float(proposal_probs[sampled].item()), sampled


def importance_weight(model_prob: float | None, proposal_prob: float | None, clip_max: float = 1.0) -> float:
    """min(clip_max, model_prob / proposal_prob), computed in log space."""
    if clip_max <= 0.0:
        return 0.0
    if model_prob is None or proposal_prob is None:
        return min(1.0, float(clip_max))
    tiny = float(np.finfo(np.float64).tiny)
    log_ratio = math.log(max(float(model_prob), tiny)) - math.log(max(float(proposal_prob), tiny))
    if log_ratio >= math.log(max(float(clip_max), 1e-12)):
        return float(clip_max)
    return float(math.exp(log_ratio))


def wdl_entropy(wdl_probs) -> float:
    """Entropy of a (win, draw, loss) distribution in nats, in [0, ln 3]."""
    entropy = 0.0
    for prob in wdl_probs:
        p = float(prob)
        if p > 0.0:
            entropy -= p * math.log(max(p, 1e-8))
    return float(entropy)


def kl_to_uniform_over_legal_moves(policy_probs: torch.Tensor, legal_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(pi || uniform over legal moves), batch mean — the negative policy entropy up to a
    per-position constant, i.e. the entropy-bonus regularizer."""
    legal = legal_mask.to(dtype=torch.float32)
    probs = policy_probs.to(dtype=torch.float32) * legal
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_uniform = -torch.log(legal.sum(dim=-1).clamp_min(1.0))
    return (probs * (torch.log(probs.clamp_min(eps)) - log_uniform.unsqueeze(-1))).sum(dim=-1).mean()


# ------------------------------------------------------------------------- game results


def is_draw_claim_now(board: chess.Board) -> bool:
    """Threefold repetition or the 50-move rule already satisfied (claimed immediately)."""
    try:
        repetition_now = board.is_repetition(3)
    except Exception:
        repetition_now = False
    return repetition_now or board.is_fifty_moves()


def is_terminal(board: chess.Board) -> bool:
    return board.is_game_over(claim_draw=False) or is_draw_claim_now(board)


def game_result(board: chess.Board) -> str:
    """'1-0', '0-1', '1/2-1/2', or '*' when the game is not over."""
    if is_draw_claim_now(board):
        return "1/2-1/2"
    if board.is_game_over(claim_draw=False):
        return board.result(claim_draw=False)
    return "*"


def wdl_from_result(board: chess.Board, result: str) -> torch.Tensor:
    """One-hot (win, draw, loss) of a finished game from the side-to-move's view."""
    if result == "1/2-1/2":
        return torch.tensor([0.0, 1.0, 0.0])
    winner = {"1-0": chess.WHITE, "0-1": chess.BLACK}[result]
    return torch.tensor([1.0, 0.0, 0.0]) if board.turn == winner else torch.tensor([0.0, 0.0, 1.0])


def wdl_to_pov(wdl_probs: torch.Tensor, source_board: chess.Board, target_board: chess.Board) -> torch.Tensor:
    """Re-express a WDL vector given from `source_board`'s side to move for `target_board`'s."""
    converted = wdl_probs.detach().to(device="cpu", dtype=torch.float32).clone()
    return converted[[2, 1, 0]] if source_board.turn != target_board.turn else converted


def expected_score(wdl_probs) -> float:
    return float(wdl_probs[0] + 0.5 * wdl_probs[1])


# --------------------------------------------------------- model, optimizer, checkpoints


def build_model(onnx_path: str, device, embedding_tensor: str = "/encoder14/ln2/betas", emb_size: int = 512):
    return model_module.Model(emb_size, lc0_embedding_onnx_path=onnx_path, lc0_embedding_tensor_name=embedding_tensor).to(device)


def parameter_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    """Split trainable parameters into backbone / final transformer block / policy head / value head."""
    policy_ids = {id(p) for p in model.move_head_model.parameters()}
    value_ids = {id(p) for p in model.value_head_model.parameters()}
    graph = model.y_lc0_embedder.model
    last_block = model_module.Lc0LastEncoderBlockFromConvertedModel(
        graph, input_node_name="encoder13_ln2_betas", output_node_name="encoder14_ln2_betas"
    )
    last_block_ids = {id(graph.get_parameter(t)) for t in last_block._parameter_key_by_target}
    for target in last_block._module_key_by_target:
        last_block_ids.update(id(p) for p in graph.get_submodule(target).parameters())

    groups = {"backbone": [], "lc0_last_block": [], "policy_head": [], "wdl_head": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in value_ids or "wdl" in name.lower():
            groups["wdl_head"].append(p)
        elif id(p) in policy_ids:
            groups["policy_head"].append(p)
        elif id(p) in last_block_ids:
            groups["lc0_last_block"].append(p)
        else:
            groups["backbone"].append(p)
    return groups


def build_optimizer(model, learning_rates: dict[str, float]) -> torch.optim.Optimizer:
    """AdamW(0.9, 0.98), no weight decay, one learning rate per parameter group."""
    param_groups = [
        {"params": params, "lr": learning_rates[name], "base_lr": learning_rates[name], "group_name": name}
        for name, params in parameter_groups(model).items()
        if params
    ]
    return torch.optim.AdamW(param_groups, lr=learning_rates["backbone"], weight_decay=0.0, betas=(0.9, 0.98))


def warmup_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup over `warmup_steps`, then constant."""
    if warmup_steps <= 0 or step > warmup_steps:
        return float(base_lr)
    return float(base_lr) * step / warmup_steps


def save_checkpoint(path: str, model, optimizer, step: int, base_model=None, ema_model=None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": int(step)}
    if base_model is not None:
        payload["base_model_state_dict"] = base_model.state_dict()
    if ema_model is not None:
        payload["ema_model_state_dict"] = ema_model.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------- warmstart


@torch.no_grad()
def warmstart(games: list[GameState], model, device, rng: random.Random, max_full_moves: int,
              temperature_sampling: bool, batch_size: int = 512) -> None:
    """Seed each game with k ~ Uniform{0..max_full_moves} full moves sampled from `model`
    (tempered by the WDL entropy when `temperature_sampling`). A game that ends inside its
    budget restarts from the initial position and spends the remaining budget."""
    was_training = model.training
    model.eval()
    targets = [2 * rng.randint(0, max_full_moves) for _ in games]
    active = [i for i, g in enumerate(games) if g.plies < targets[i]]
    while active:
        next_active = []
        for start in range(0, len(active), batch_size):
            chunk = active[start : start + batch_size]
            x, mask, meta, legal_moves, legal_indices = build_batch([games[i].board for i in chunk], device)
            logits, _, aux = model(x, lc0_meta=meta, return_lc0_wdl=True)
            for row, i in enumerate(chunk):
                game = games[i]
                temperature = wdl_entropy(aux["lc0_wdl_probs"][row].tolist()) if temperature_sampling else None
                move, _, _, _ = sample_move(logits[row], legal_moves[row], legal_indices[row],
                                            promotion_offsets_for_row(aux, row), temperature)
                if move is None:
                    continue
                game.board.push(move)
                game.plies += 1
                if is_terminal(game.board):
                    targets[i] -= game.plies
                    games[i] = GameState.new()
                if games[i].plies < targets[i]:
                    next_active.append(i)
        active = next_active
    if was_training:
        model.train()

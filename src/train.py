"""Self-play fine-tuning of a searchless chess network with a prior-directed exploration term.

One optimizer step per batch: every parallel game advances one ply, sampled from
mu(.|s) ~ pi_theta(.|s)^(1/tau(s)) with tau(s) the WDL entropy of the value head, and
each transition is consumed once with

    rho  = min(1, pi_theta(a|s) / mu(a|s))
    A    = (1 - v_ema(s')) - v_theta(s)
    loss = -mean(rho * A * log pi_theta(a|s))
           + beta * KL(pi_base || pi_theta)          (or reverse KL / KL to uniform)
           + c_v * mean(rho * CE(flip(wdl_ema(s')), wdl_theta(s)))
           + c_v * w_base * KL(wdl_base(s) || wdl_theta(s))

after which the EMA network tracks theta. Finished games restart from the initial position.

Networks: `pi_base` (frozen reference), `pi_theta` (trained), `pi_ema` (slow copy for value
targets). Run `python src/train.py --help` for the knobs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import chess
import numpy as np
import torch
import torch.nn.functional as F

import selfplay as sp
from chess_encoding import get_move_index


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", default="model_assets/BT4-1024x15x32h-swa-6147500_with_embedding.onnx")
    p.add_argument("--embedding-tensor", default="/encoder14/ln2/betas")
    p.add_argument("--emb-size", type=int, default=512)
    p.add_argument("--num-games", type=int, default=1024, help="Parallel games = positions per optimizer step.")
    p.add_argument("--minibatch-size", type=int, default=32, help="Gradient micro-batch; accumulated over the step.")
    p.add_argument("--forward-batch-size", type=int, default=512, help="Chunk size for no-grad forwards.")
    p.add_argument("--total-gradient-steps", type=int, default=2000)
    p.add_argument("--regularizer", choices=["forward_kl", "reverse_kl", "entropy", "none"], default="forward_kl")
    p.add_argument("--beta", type=float, default=0.01, help="Weight of the policy regularizer.")
    p.add_argument("--sample-with-wdl-entropy-temperature", action="store_true",
                   help="Sample moves from pi^(1/tau) with tau = WDL entropy; otherwise from pi.")
    p.add_argument("--sampling-temperature-wdl-source", choices=["online", "ema"], default="online",
                   help="Whose WDL entropy sets tau: the trained network or the EMA copy.")
    p.add_argument("--wdl-kl-weight", type=float, default=0.2, help="c_v")
    p.add_argument("--base-wdl-kl-weight", type=float, default=0.1, help="w_base (0 disables the term)")
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--lr", type=float, default=3e-6, help="Backbone learning rate.")
    p.add_argument("--lc0-last-block-lr", type=float, default=1e-5)
    p.add_argument("--policy-head-lr", type=float, default=1e-5)
    p.add_argument("--wdl-head-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-plies", type=int, default=512, help="Games are cut (and restarted) at this length.")
    p.add_argument("--warmstart-max-full-moves", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="output/train")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


@dataclass
class Transition:
    pre: chess.Board
    post: chess.Board
    action: int  # slot of the sampled move in the 4096-slot policy
    rho: float  # min(1, pi/mu)
    wdl_online_pre: torch.Tensor  # wdl_theta(s), mover's view
    result: str  # "*" unless s' ended the game
    wdl_target: torch.Tensor | None = None  # flip(wdl_ema(s')) or the one-hot result, in s's view
    advantage: float | None = None
    base_policy: torch.Tensor | None = None  # pi_base(.|s) over the 4096 slots
    base_wdl: torch.Tensor | None = None
    restarted: bool = False  # the game was restarted after this ply (ended or hit --max-plies)


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def forward(model, boards, device):
    """Forward `boards`; returns masked logits, legal mask, WDL probs, raw logits, aux, legal moves, legal slots."""
    x, mask, meta, legal_moves, legal_indices = sp.build_batch(boards, device)
    logits, _, aux = model(x, lc0_meta=meta, return_lc0_wdl=True)
    masked = sp.masked_logits(logits, mask, legal_moves, legal_indices, aux)
    return masked, mask, aux["lc0_wdl_probs"], logits, aux, legal_moves, legal_indices


def reverse_kl(log_p, q, eps=1e-8):
    p = log_p.exp().clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q.clamp_min(eps)))).sum(dim=-1).mean()


def step_loss(masked_logits, mask, wdl, target_idx, rho, advantage, wdl_target, base_policy, base_wdl, args):
    """The loss of the module docstring for one minibatch; every target is a constant."""
    log_policy = torch.log_softmax(masked_logits, dim=-1)
    policy_loss = (F.cross_entropy(masked_logits, target_idx, reduction="none") * rho * advantage).mean()

    log_wdl = torch.log(wdl.clamp_min(1e-8))
    wdl_loss = (F.kl_div(log_wdl, wdl_target, reduction="none").sum(dim=-1) * rho).mean()

    if args.regularizer == "forward_kl":
        reg = F.kl_div(log_policy, base_policy, reduction="batchmean")
    elif args.regularizer == "reverse_kl":
        reg = reverse_kl(log_policy, base_policy)
    elif args.regularizer == "entropy":
        reg = sp.kl_to_uniform_over_legal_moves(log_policy.exp(), mask)
    else:
        reg = torch.zeros((), device=masked_logits.device)

    if args.base_wdl_kl_weight > 0.0:
        if args.regularizer == "reverse_kl":
            base_wdl_term = reverse_kl(log_wdl, base_wdl)
        else:
            base_wdl_term = F.kl_div(log_wdl, base_wdl, reduction="batchmean")
    else:
        base_wdl_term = torch.zeros((), device=masked_logits.device)

    loss = (
        policy_loss
        + args.beta * reg
        + args.wdl_kl_weight * wdl_loss
        + args.base_wdl_kl_weight * args.wdl_kl_weight * base_wdl_term
    )
    return loss, {"policy": policy_loss, "reg": reg, "wdl": wdl_loss, "base_wdl": base_wdl_term}


@torch.no_grad()
def sample_transitions(games, trainable, ema, args, device):
    """Advance every game one ply from mu; record rho and wdl_theta(s); restart finished games."""
    trainable.eval()
    transitions = []
    for game_chunk in chunks(list(range(len(games))), args.forward_batch_size):
        boards = [games[i].board for i in game_chunk]
        _, _, wdl, logits, aux, legal_moves, legal_indices = forward(trainable, boards, device)
        tau_wdl = wdl
        if args.sample_with_wdl_entropy_temperature and args.sampling_temperature_wdl_source == "ema":
            tau_wdl = forward(ema, boards, device)[2]
        for row, game_idx in enumerate(game_chunk):
            game = games[game_idx]
            tau = sp.wdl_entropy(tau_wdl[row].tolist()) if args.sample_with_wdl_entropy_temperature else None
            move, p_model, p_proposal, _ = sp.sample_move(
                logits[row], legal_moves[row], legal_indices[row], sp.promotion_offsets_for_row(aux, row), tau
            )
            if move is None:
                games[game_idx] = sp.GameState.new()
                continue
            pre = game.board.copy(stack=True)
            game.board.push(move)
            game.plies += 1
            post = game.board.copy(stack=True)
            result = sp.game_result(post)
            restart = result != "*" or game.plies >= args.max_plies
            transitions.append(Transition(
                pre=pre, post=post,
                action=int(get_move_index(move.uci(), pre.turn == chess.BLACK, board=pre)),
                rho=sp.importance_weight(p_model, p_proposal, clip_max=1.0),
                wdl_online_pre=wdl[row].detach().cpu().float().clone(),
                result=result,
                restarted=restart,
            ))
            if restart:
                games[game_idx] = sp.GameState.new()
    trainable.train()
    return transitions


@torch.no_grad()
def label_transitions(transitions, base, ema, args, device):
    """EMA bootstrap at s' (one-hot when the game ended), the advantage, and pi_base / wdl_base at s."""
    pending = [t for t in transitions if t.result == "*"]
    for chunk in chunks(pending, args.forward_batch_size):
        wdl_post = forward(ema, [t.post for t in chunk], device)[2].cpu().float()
        for t, w in zip(chunk, wdl_post):
            t.wdl_target = sp.wdl_to_pov(w, source_board=t.post, target_board=t.pre)
    for t in transitions:
        if t.result != "*":
            t.wdl_target = sp.wdl_to_pov(sp.wdl_from_result(t.post, t.result), source_board=t.post, target_board=t.pre)
        t.advantage = sp.expected_score(t.wdl_target) - sp.expected_score(t.wdl_online_pre)
    for chunk in chunks(transitions, args.forward_batch_size):
        masked, _, wdl_base, _, _, _, _ = forward(base, [t.pre for t in chunk], device)
        for t, pol, w in zip(chunk, torch.softmax(masked, dim=-1).cpu().float(), wdl_base.cpu().float()):
            t.base_policy, t.base_wdl = pol, w


def optimizer_step(transitions, trainable, optimizer, args, device) -> dict[str, float]:
    """One AdamW step on the whole batch, accumulated over minibatches. Returns mean loss terms."""
    optimizer.zero_grad(set_to_none=True)
    num_chunks = max(1, math.ceil(len(transitions) / args.minibatch_size))
    metrics: dict[str, float] = {}
    for chunk in chunks(transitions, args.minibatch_size):
        masked, mask, wdl, _, _, _, _ = forward(trainable, [t.pre for t in chunk], device)
        loss, parts = step_loss(
            masked, mask, wdl,
            target_idx=torch.tensor([t.action for t in chunk], dtype=torch.long, device=device),
            rho=torch.tensor([t.rho for t in chunk], dtype=torch.float32, device=device),
            advantage=torch.tensor([t.advantage for t in chunk], dtype=torch.float32, device=device),
            wdl_target=torch.stack([t.wdl_target for t in chunk]).to(device),
            base_policy=torch.stack([t.base_policy for t in chunk]).to(device),
            base_wdl=torch.stack([t.base_wdl for t in chunk]).to(device),
            args=args,
        )
        (loss / num_chunks).backward()
        for name, value in {"loss": loss, **parts}.items():
            metrics[name] = metrics.get(name, 0.0) + float(value.item()) / num_chunks
    optimizer.step()
    return metrics


@torch.no_grad()
def update_ema(ema, trainable, decay):
    for ema_p, p in zip(ema.parameters(), trainable.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for ema_b, b in zip(ema.buffers(), trainable.buffers()):
        ema_b.copy_(b.detach())


def batch_metrics(transitions) -> dict[str, float]:
    rho = np.array([t.rho for t in transitions])
    adv = np.array([t.advantage for t in transitions])
    return {
        "positions": len(transitions),
        "finished_games": sum(t.result != "*" for t in transitions),
        "restarted_games": sum(t.restarted for t in transitions),
        "rho_mean": float(rho.mean()),
        "advantage_mean": float(adv.mean()),
        "advantage_abs_mean": float(np.abs(adv).mean()),
    }


class MetricsLog:
    """Appends one JSON object per step to `metrics.jsonl` and prints a compact line."""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, "metrics.jsonl")

    def write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        shown = {k: v for k, v in record.items() if k in ("step", "loss", "policy", "reg", "wdl", "finished_games")}
        print(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in shown.items()), flush=True)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    trainable = sp.build_model(args.onnx, device, args.embedding_tensor, args.emb_size)
    base = sp.build_model(args.onnx, device, args.embedding_tensor, args.emb_size)
    ema = sp.build_model(args.onnx, device, args.embedding_tensor, args.emb_size)
    base.load_state_dict(trainable.state_dict())
    ema.load_state_dict(trainable.state_dict())
    for frozen in (base, ema):
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad = False
    optimizer = sp.build_optimizer(trainable, {
        "backbone": args.lr, "lc0_last_block": args.lc0_last_block_lr,
        "policy_head": args.policy_head_lr, "wdl_head": args.wdl_head_lr,
    })
    log = MetricsLog(args.output_dir)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    games = [sp.GameState.new() for _ in range(args.num_games)]
    sp.warmstart(games, trainable, device, random.Random(args.seed), args.warmstart_max_full_moves,
                 args.sample_with_wdl_entropy_temperature, args.forward_batch_size)

    for step in range(1, args.total_gradient_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = sp.warmup_lr(step, group["base_lr"], args.warmup_steps)
        transitions = sample_transitions(games, trainable, ema, args, device)
        label_transitions(transitions, base, ema, args, device)
        losses = optimizer_step(transitions, trainable, optimizer, args, device)
        update_ema(ema, trainable, args.ema_decay)
        log.write({"step": step, **losses, **batch_metrics(transitions)})
        if args.save_every > 0 and (step % args.save_every == 0 or step == args.total_gradient_steps):
            sp.save_checkpoint(os.path.join(args.output_dir, "model.pt"), trainable, optimizer, step, base_model=base, ema_model=ema)


if __name__ == "__main__":
    main()

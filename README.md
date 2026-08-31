# Beyond Search-Imitation: Prior-Directed Exploration for Searchless Chess

Code for the paper [*"Beyond Search-Imitation: Prior-Directed Exploration for
Searchless Chess"*](https://arxiv.org/abs/2608.27757) (arXiv:2608.27757).

A searchless chess network — Lc0's supervised-distilled BT4 Chessformer — is
fine-tuned for single-pass playing strength by self-play policy gradient. The usual
entropy bonus is replaced by a forward (mass-covering) KL toward the network's own
MCTS-derived prior, and moves are sampled at a temperature set by the value head's
own uncertainty (its win/draw/loss entropy).

```
src/
  train.py            the training loop — one screen of algorithm, see its docstring
  selfplay.py         batching, move sampling, importance weights, game results,
                      WDL bookkeeping, optimizer groups, checkpoints, warmstart
  network.py          the network: the converted Lc0 trunk + policy and value heads
  chess_encoding.py   board planes, Lc0 metadata, move indexing
  network_prep.py     converts the Lc0 network file to the ONNX asset the network loads
scripts/              data + network preparation
data/                 the paper's evaluation suites and opening books, vendored compressed
tests/                closed-form checks of every loss term and helper, a one-step
                      end-to-end run
```

## Setup

Python 3.10+ and a CUDA GPU for training at the paper's batch size (three copies of
the 191M-parameter network stay resident; small runs also work on CPU).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/prepare_data.sh       # unpack + checksum-verify the vendored data assets
```

Prepare the base network (requires an `lc0` binary, **v0.32 or newer**, for
`leela2onnx` — with v0.32.1 the conversion reproduces the paper's ONNX asset
byte-for-byte, while older builds emit a different graph that the network module
cannot load; the `BT4-1024x15x32h-swa-6147500` network file is distributed by the
Lc0 project):

```bash
python scripts/prepare_network.py \
  --input /path/to/BT4-1024x15x32h-swa-6147500.pb.gz \
  --lc0-bin /path/to/lc0
```

This writes `model_assets/BT4-1024x15x32h-swa-6147500_with_embedding.onnx`, which
the trainer loads.

## Training

```bash
python src/train.py --sample-with-wdl-entropy-temperature \
    --regularizer forward_kl --beta 0.01 --output-dir output/forward_kl
```

Each optimizer step advances 1,024 parallel games by one ply, sampled from the
tempered policy, and consumes those transitions once (`--help` lists every knob;
the defaults are the paper's Table 2). The exploration term is selected with
`--regularizer {forward_kl,reverse_kl,entropy,none}` and weighted by `--beta`;
`--sample-with-wdl-entropy-temperature` turns the entropy-adaptive temperature on.
The run writes `config.json`, one JSON record per step to `metrics.jsonl` (loss
terms, importance-weight and advantage statistics, finished games), and `model.pt`
with the trained, EMA and base weights.

`train.py` is written to be read: the loss is stated in its docstring and computed
in `step_loss`, the three passes of a step (`sample_transitions`,
`label_transitions`, `optimizer_step`) are top-level functions, and every helper it
calls lives in `selfplay.py`.

## Data

`data/` vendors, gzip-compressed with SHA-256 manifests (`data/SHA256SUMS`), the
exact evaluation suites of the paper — not just the recipe that sampled them:

- `data/test/puzzles.csv` — the published 10k tactics suite of Ruoss et al. (2024),
  used for β selection.
- `data/test/puzzles100k.csv` + `elo2400_20k_suite.csv` — the paper's 100k uniform
  and ≥2400-rated 20k suites, with their retrieved real-history files
  (`*_history.jsonl`) and multi-solution mate ground truth (`*_mate_trees.jsonl`).
- `data/test/mate_suite.csv` / `mate_in5_*` — the mate-in-1..5 suite, its history,
  and its acceptable-move sets.
- `data/test/puzzle_onehot_trainset_v2.csv` — the exact training pool of the
  supervised puzzle-tuned control.
- `data/openings/UHO_Lichess_4852_v1.epd` — the UHO opening book (Stefan Pohl)
  used for arena games, plus the ECO-by-ply lookup in
  `data/openings/eco_lichess_by_plies/`.

Puzzle data derives from the [Lichess puzzle database](https://database.lichess.org/)
(CC0). The full DB (~1 GB; the paper used an April 2026 snapshot) is **not**
vendored — it is needed only to re-sample suites from scratch. Trained checkpoints
are not part of this repository.

## Tests

```bash
python -m pytest tests -q
```

`tests/test_train.py` runs the trainer for one step on CPU; it skips automatically
when the network file is absent.

## License

MIT — see `LICENSE`. Vendored data derives from CC0 Lichess data and the freely
distributed UHO opening book; the BT4 network itself is distributed by the Lc0
project under its own terms.

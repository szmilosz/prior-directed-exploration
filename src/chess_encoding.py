from __future__ import annotations

import csv

import chess
import numpy as np
import torch


def get_lc0_meta_from_board(board: chess.Board, include_repetition: bool = False) -> dict[str, object]:
    side_to_move = 0 if board.turn else 1
    castling_rights = board.castling_rights
    if side_to_move == 0:
        us_ooo, us_oo = (castling_rights >> chess.A1) & 1, (castling_rights >> chess.H1) & 1
        them_ooo, them_oo = (castling_rights >> chess.A8) & 1, (castling_rights >> chess.H8) & 1
    else:
        us_ooo, us_oo = (castling_rights >> chess.A8) & 1, (castling_rights >> chess.H8) & 1
        them_ooo, them_oo = (castling_rights >> chess.A1) & 1, (castling_rights >> chess.H1) & 1

    repetition_flag = 0.0
    if include_repetition:
        try:
            repetition_flag = 1.0 if board.is_repetition(2) else 0.0
        except Exception:
            repetition_flag = 0.0

    history_repetition = np.zeros((8,), dtype=np.float32)
    history_repetition[0] = repetition_flag

    return {
        "castling": np.array([us_ooo, us_oo, them_ooo, them_oo], dtype=np.float32),
        "side_to_move": float(side_to_move),
        "rule50_count": float(max(0, min(255, board.halfmove_clock))),
        "history_repetition": history_repetition,
    }


def build_lc0_meta_batch(meta_items: list[dict[str, object]] | tuple[dict[str, object], ...] | None):
    if not meta_items:
        return None

    castling = torch.tensor(np.stack([m["castling"] for m in meta_items], axis=0), dtype=torch.float32)
    side_to_move = torch.tensor([m["side_to_move"] for m in meta_items], dtype=torch.float32)
    rule50_count = torch.tensor([m["rule50_count"] for m in meta_items], dtype=torch.float32)
    history_repetition = torch.tensor(
        np.stack([m["history_repetition"] for m in meta_items], axis=0),
        dtype=torch.float32,
    )
    return {
        "castling": castling,
        "side_to_move": side_to_move,
        "rule50_count": rule50_count,
        "history_repetition": history_repetition,
    }


def build_lc0_meta_batch_from_boards(
    boards: list[chess.Board] | tuple[chess.Board, ...],
    include_repetition: bool = False,
):
    meta_items = [get_lc0_meta_from_board(board, include_repetition=include_repetition) for board in boards]
    return build_lc0_meta_batch(meta_items)


def get_input_tensor(board: chess.Board) -> torch.Tensor:
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, board.turn):
            idx = np.unravel_index(square, (8, 8))
            board_tensor[piece - 1][7 - idx[0]][idx[1]] = 1.0
        for square in board.pieces(piece, not board.turn):
            idx = np.unravel_index(square, (8, 8))
            board_tensor[piece + 5][7 - idx[0]][idx[1]] = 1.0

    if board.turn == chess.BLACK:
        board_tensor = np.flip(board_tensor, axis=1).copy()
    return torch.tensor(board_tensor, dtype=torch.float32)


def convert_square_index(square: int, flip: bool) -> int:
    rank = square // 8
    file_idx = square % 8
    if not flip:
        new_rank = 7 - rank
        return new_rank * 8 + file_idx
    return rank * 8 + file_idx


def get_move_index(move: str, flip: bool, board: chess.Board | None = None) -> int:
    from_square = chess.parse_square(move[:2])
    to_square = chess.parse_square(move[2:4])

    if board is not None:
        piece = board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.KING:
            castling_remap = {
                (chess.E1, chess.G1): (chess.E1, chess.H1),
                (chess.E1, chess.C1): (chess.E1, chess.A1),
                (chess.E8, chess.G8): (chess.E8, chess.H8),
                (chess.E8, chess.C8): (chess.E8, chess.A8),
            }
            remapped = castling_remap.get((from_square, to_square))
            if remapped is not None:
                from_square, to_square = remapped

    from_square = convert_square_index(from_square, flip)
    to_square = convert_square_index(to_square, flip)
    return from_square * 64 + to_square


def load_puzzle_entries(puzzle_csv_path):
    puzzle_entries = []
    with open(puzzle_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            moves_text = (row.get("Moves") or "").strip()
            fen = (row.get("FEN") or "").strip()
            move_tokens = [m for m in moves_text.split() if m]

            entry = {
                "puzzle_id": (row.get("PuzzleId") or f"row_{row_idx}"),
                "samples": [],
                "valid": True,
            }

            if (not fen) or len(move_tokens) < 2:
                entry["valid"] = False
                puzzle_entries.append(entry)
                continue

            try:
                board = chess.Board(fen)

                # The first move is from the opponent side in this setup.
                first_move = chess.Move.from_uci(move_tokens[0])
                if first_move not in board.legal_moves:
                    raise ValueError("illegal first move")
                board.push(first_move)

                # Evaluate only our moves (odd indices), while auto-playing continuation.
                for idx in range(1, len(move_tokens), 2):
                    our_move = chess.Move.from_uci(move_tokens[idx])
                    if our_move not in board.legal_moves:
                        raise ValueError("illegal our move")

                    entry["samples"].append((board.fen(), our_move.uci()))

                    # Apply the ground-truth move to reach the puzzle continuation.
                    board.push(our_move)

                    if idx + 1 < len(move_tokens):
                        opp_move = chess.Move.from_uci(move_tokens[idx + 1])
                        if opp_move not in board.legal_moves:
                            raise ValueError("illegal opponent continuation move")
                        board.push(opp_move)
            except Exception:
                entry["valid"] = False
                entry["samples"] = []

            puzzle_entries.append(entry)

    return puzzle_entries
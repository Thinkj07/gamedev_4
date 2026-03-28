from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

from board import Board
from ai import RandomAI
from constants import ROWS, COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_FILE = "connect4_data.csv"
DEFAULT_GAMES = 5_000
_CELL_HEADERS = [f"cell_{i}" for i in range(ROWS * COLS)]   # 56 columns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(board: Board) -> List[int]:
    """Return the 7×8 grid as a 56-element row-major list."""
    flat: List[int] = []
    for row in board.grid:
        flat.extend(row)
    return flat


def _next_player(p: int, num_p: int) -> int:
    return p % num_p + 1


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_game(
    num_p: int,
    ai: RandomAI,
) -> Tuple[List[List[int]], int]:
    """
    Play one complete game with *num_p* random players.

    Returns
    -------
    states : list of 56-element snapshots taken **after** each move
    winner : 0 (draw) or 1–3 (winning player id)
    """
    board = Board()
    states: List[List[int]] = []
    current = 1 

    while True:
        col = ai.get_move(board, current, num_p)
        if col == -1:
            break  
        r = board.drop(col, current)
        if r < 0:
            break  
        states.append(_flatten(board))
        if board.check_win_at(r, col):
            return states, current 
        if board.is_full():
            return states, 0 
        current = _next_player(current, num_p)

    return states, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Connect Four training data.")
    parser.add_argument(
        "--games", type=int, default=DEFAULT_GAMES,
        help=f"Number of games to simulate (default: {DEFAULT_GAMES})"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE,
        help=f"Output CSV file path (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: None)"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ai = RandomAI()
    output_path = Path(args.output)

    total_games = args.games
    total_states = 0
    start_time = time.time()

    print(f"[Data Generator] Starting {total_games:,} games → {output_path}")
    print(f"  Board: {ROWS} rows × {COLS} cols  |  Players: 2 or 3 (random per game)")
    print(f"  Seed : {args.seed}")
    print()

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CELL_HEADERS + ["num_p", "winner"])

        for game_idx in range(total_games):
            num_p = random.choice([2, 3])

            states, winner = simulate_game(num_p, ai)

            for state in states:
                writer.writerow(state + [num_p, winner])

            total_states += len(states)
            if (game_idx + 1) % 500 == 0 or game_idx == total_games - 1:
                elapsed = time.time() - start_time
                pct = (game_idx + 1) / total_games * 100
                rate = (game_idx + 1) / max(elapsed, 1e-9)
                print(
                    f"  Game {game_idx + 1:>6,}/{total_games:,}  "
                    f"({pct:5.1f}%)  "
                    f"states so far: {total_states:>9,}  "
                    f"rate: {rate:6.1f} games/s  "
                    f"elapsed: {elapsed:6.1f}s"
                )
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print()
    print(f"[Done] {total_states:,} state rows written to '{output_path}'")
    print(f"       Total time: {elapsed:.1f}s  |  Avg states/game: {total_states/total_games:.1f}")


if __name__ == "__main__":
    main()

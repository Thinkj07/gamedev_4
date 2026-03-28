from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from board import Board
from constants import ROWS, COLS

# ---------------------------------------------------------------------------
# Re-export helpers from the original ai module so callers only need ai_ml
# ---------------------------------------------------------------------------
from ai import (
    RandomAI,
    _next_player,
    _valid_moves,
    _evaluate_board,   # fallback
    _WIN,
    _LOSS,
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "connect4_ml_model.pkl"

_model = None          # sklearn Pipeline (StandardScaler + MLPClassifier)
_model_loaded = False  # True even when model file is missing (avoids retrying)


def _load_model():
    """Load the sklearn pipeline once; warn and fall back if absent."""
    global _model, _model_loaded
    if _model_loaded:
        return
    _model_loaded = True
    if MODEL_PATH.exists():
        try:
            import joblib
            _model = joblib.load(MODEL_PATH)
            dummy = np.zeros((1, 57), dtype=np.float32)
            _model.predict_proba(dummy)
            print(f"[ai_ml] Model loaded from '{MODEL_PATH}'")
        except Exception as exc:
            warnings.warn(
                f"[ai_ml] Failed to load model from '{MODEL_PATH}': {exc}\n"
                "         Falling back to hand-crafted heuristic.",
                RuntimeWarning, stacklevel=3,
            )
            _model = None
    else:
        warnings.warn(
            f"[ai_ml] Model file not found: '{MODEL_PATH}'.\n"
            "         Run train_model.py first.\n"
            "         Falling back to hand-crafted heuristic / random rollout.",
            RuntimeWarning, stacklevel=3,
        )

_load_model()

# --------------------------------------------------------------------------
# Core ML helper
# ---------------------------------------------------------------------------

def _flatten_board(board: Board, num_p: int) -> np.ndarray:
    """
    Build the 57-element feature vector expected by the model.

    Layout: [cell_0 … cell_55, num_p]
    row-major (row 0 first, then row 1, …)
    """
    flat = np.empty(57, dtype=np.float32)
    idx = 0
    for row in board.grid:
        for val in row:
            flat[idx] = val
            idx += 1
    flat[56] = num_p
    return flat.reshape(1, -1)


def _evaluate_board_ml(board: Board, root: int, num_p: int) -> float:
    """
    Paranoid board evaluation using the ML model.

    Returns
    -------
    float in [-1.0, 1.0]
        P(root wins) - max(P(opponent wins))

    Falls back to the hand-crafted heuristic (normalised to [-1, 1]) if the
    model is unavailable.
    """
    if _model is None:
        raw: int = _evaluate_board(board, root, num_p)
        return math.tanh(raw / 1000.0)

    feat = _flatten_board(board, num_p)
    proba = _model.predict_proba(feat)[0]          
    classes: list = list(_model.classes_)           

    def _p(player: int) -> float:
        try:
            return float(proba[classes.index(player)])
        except ValueError:
            return 0.0

    p_root = _p(root)
    opponents = [pl for pl in range(1, num_p + 1) if pl != root]
    p_best_opp = max((_p(opp) for opp in opponents), default=0.0)

    return p_root - p_best_opp          # in [−1, 1]

# ---------------------------------------------------------------------------
# ML_MinimaxAI
# ---------------------------------------------------------------------------

class ML_MinimaxAI:
    """
    Paranoid alpha-beta Minimax that uses the ML model at leaf nodes instead
    of the hand-crafted window heuristic.

    The overall tree structure, pruning logic, and terminal detection are
    identical to MinimaxAI in ai.py.  Only the depth-0 evaluation changes.
    """

    __slots__ = ("depth",)

    def __init__(self, depth: int = 4):
        """
        Parameters
        ----------
        depth : int
            Search depth (number of half-moves / plies). Values 2-6 are
            practical; higher values are slower but stronger.
        """
        self.depth = max(2, min(6, int(depth)))

    def get_move(self, board: Board, cur: int, num_p: int) -> int:
        """Return the best column for player *cur* to play."""
        moves = _valid_moves(board)
        if not moves:
            return -1
        if len(moves) == 1:
            return moves[0]

        best_col = moves[0]
        best_val = -math.inf
        alpha, beta = -math.inf, math.inf

        for col in moves:
            r = board.drop(col, cur)
            if r < 0:
                continue
            if board.check_win_at(r, col):
                board.undo(col)
                return col
            nxt = _next_player(cur, num_p)
            v, _ = self._search(board, self.depth - 1, alpha, beta,
                                nxt, cur, num_p, nxt == cur)
            board.undo(col)
            if v > best_val:
                best_val = v
                best_col = col
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break

        return best_col

    def _search(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        to_move: int,
        root: int,
        num_p: int,
        maximizing: bool,
    ) -> Tuple[float, Optional[int]]:
        """Recursive alpha-beta with ML leaf evaluation."""
        if board.is_full():
            return 0.0, None
        if depth == 0:
            score = _evaluate_board_ml(board, root, num_p)
            return score, None
        moves = _valid_moves(board)
        if not moves:
            return 0.0, None
        if maximizing:
            best = -math.inf
            best_c: Optional[int] = None
            for col in moves:
                r = board.drop(col, to_move)
                if r < 0:
                    continue
                win = board.check_win_at(r, col)
                if win == root:
                    board.undo(col)
                    return 1.0, col          
                if win and win != root:
                    board.undo(col)
                    return -1.0, col         
                nxt = _next_player(to_move, num_p)
                v, _ = self._search(board, depth - 1, alpha, beta,
                                    nxt, root, num_p, nxt == root)
                board.undo(col)
                if v > best:
                    best = v
                    best_c = col
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best, best_c

        best = math.inf
        best_c = None
        for col in moves:
            r = board.drop(col, to_move)
            if r < 0:
                continue
            win = board.check_win_at(r, col)
            if win:
                board.undo(col)
                return -1.0, col
            nxt = _next_player(to_move, num_p)
            v, _ = self._search(board, depth - 1, alpha, beta,
                                 nxt, root, num_p, nxt == root)
            board.undo(col)
            if v < best:
                best = v
                best_c = col
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best, best_c

# ---------------------------------------------------------------------------
# ML_MCTSAI
# ---------------------------------------------------------------------------

class _MCTSNode:
    """UCT tree node.  Stores statistics for the player who *made* the move."""

    __slots__ = ("parent", "move", "who", "children", "_untried", "N", "W")

    def __init__(self, parent: Optional["_MCTSNode"], move: int, who: Optional[int]):
        self.parent = parent
        self.move = move
        self.who = who                                # player who played here
        self.children: dict[int, _MCTSNode] = {}
        self._untried: Optional[List[int]] = None
        self.N: int = 0
        self.W: float = 0.0


class ML_MCTSAI:
    """
    Monte Carlo Tree Search (UCT) that replaces the random rollout with a
    single ML board evaluation at each leaf expansion.

    After expansion, the board state is passed to _evaluate_board_ml which
    returns a paranoid score in [-1, 1].  This score is used directly as the
    W (win-value) contribution during back-propagation, eliminating the
    expensive random play-out loop.

    The key insight: instead of simulating to a terminal state (expensive),
    the model acts as a learned value function that estimates win probability
    from any intermediate position.
    """

    __slots__ = ("simulations", "explore_c")

    def __init__(self, simulations: int = 1000, explore_c: float = math.sqrt(2.0)):
        """
        Parameters
        ----------
        simulations : int
            Number of MCTS iterations (200-5000).
        explore_c : float
            UCB exploration constant (default √2 ≈ 1.414).
        """
        self.simulations = max(200, min(5000, int(simulations)))
        self.explore_c = explore_c

    def get_move(self, board: Board, cur: int, num_p: int) -> int:
        """Return the best column for player *cur* to play."""
        moves = _valid_moves(board)
        if not moves:
            return -1
        if len(moves) == 1:
            return moves[0]
        for col in moves:
            b = board.copy()
            r = b.drop(col, cur)
            if r >= 0 and b.check_win_at(r, col) == cur:
                return col

        root_node = _MCTSNode(None, -1, None)
        root_node._untried = list(moves)

        for _ in range(self.simulations):
            b = board.copy()
            path: List[_MCTSNode] = []
            node = root_node
            to_move = cur

            # ---- Selection + Expansion ----
            while True:
                w = b.winner()
                if w != 0 or b.is_full():
                    self._backprop_winner(path, w, cur, num_p)
                    break
                if node._untried:
                    col = node._untried.pop()
                    r = b.drop(col, to_move)
                    if r < 0:
                        continue     
                    child = _MCTSNode(node, col, to_move)
                    vm = _valid_moves(b)
                    child._untried = list(vm) if vm else []
                    node.children[col] = child
                    path.append(child)
                    to_move = _next_player(to_move, num_p)
                    win_here = b.check_win_at(r, col)
                    if win_here:
                        self._backprop_winner(path, win_here, cur, num_p)
                    else:
                        ml_score = _evaluate_board_ml(b, cur, num_p)
                        self._backprop_score(path, ml_score)
                    break
                if not node.children:
                    ml_score = _evaluate_board_ml(b, cur, num_p)
                    self._backprop_score(path, ml_score)
                    break
                ch = self._uct_select(node)
                r = b.drop(ch.move, to_move)
                if r < 0:
                    ml_score = _evaluate_board_ml(b, cur, num_p)
                    self._backprop_score(path, ml_score)
                    break
                path.append(ch)
                to_move = _next_player(to_move, num_p)
                node = ch

        if not root_node.children:
            return random.choice(moves)

        best_col = max(
            root_node.children.keys(),
            key=lambda c: (root_node.children[c].N, random.random()),
        )
        return best_col

    # ------------------------------------------------------------------
    # UCT child selection
    # ------------------------------------------------------------------

    def _uct_select(self, node: _MCTSNode) -> _MCTSNode:
        """Return the child with the highest UCB1 score."""
        children = list(node.children.values())
        total_n = sum(ch.N for ch in children)
        log_n = math.log(total_n + 1e-9)
        best_nodes: List[_MCTSNode] = []
        best_s = -math.inf
        for ch in children:
            if ch.N == 0:
                ucb = math.inf
            else:
                exploit = ch.W / ch.N
                explore = self.explore_c * math.sqrt(log_n / ch.N)
                ucb = exploit + explore
            if ucb > best_s:
                best_s = ucb
                best_nodes = [ch]
            elif math.isclose(ucb, best_s) or (ucb == math.inf == best_s):
                best_nodes.append(ch)
        return random.choice(best_nodes)

    # ------------------------------------------------------------------
    # Back-propagation – two variants
    # ------------------------------------------------------------------

    def _backprop_winner(
        self,
        path: List[_MCTSNode],
        winner: int,
        root_player: int,
        num_p: int,
    ) -> None:
        """
        Back-propagate an exact terminal result.

        W += 1.0  for each node whose player is the winner
        W += 0.0  for draws or opponents
        """
        for node in path:
            node.N += 1
            if winner != 0 and node.who == winner:
                node.W += 1.0

    def _backprop_score(self, path: List[_MCTSNode], ml_score: float) -> None:
        """
        Back-propagate a continuous paranoid score from _evaluate_board_ml.

        ml_score ∈ [-1, 1]  (positive = good for root_player)

        We map to W using a sigmoid-like shift to keep W/N ∈ (0, 1)
        so UCB arithmetic stays meaningful:

            W_contrib = (ml_score + 1) / 2    ∈ [0, 1]

        A node whose player is the root benefits from a positive score;
        all nodes get the same contribution because the paranoid score already
        encodes coalition reasoning (root vs strongest opponent).
        """
        w_contrib = (ml_score + 1.0) / 2.0   # map [−1,1] → [0,1]
        for node in path:
            node.N += 1
            node.W += w_contrib

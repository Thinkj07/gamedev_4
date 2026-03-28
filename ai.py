"""AI players: Random, MCTS (UCT), and paranoid Minimax for 2–3 player Connect Four."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from board import Board
from constants import COLS, ROWS


def _next_player(p: int, num_p: int) -> int:
    return p % num_p + 1


def _valid_moves(board: Board) -> List[int]:
    return board.valid_moves()

def _iter_windows() -> Sequence[Tuple[Tuple[int, int], ...]]:
    """All axis-aligned / diagonal segments of length 4."""
    out: List[Tuple[Tuple[int, int], ...]] = []
    for r in range(ROWS):
        for c in range(COLS - 3):
            out.append(tuple((r, c + k) for k in range(4)))
    for c in range(COLS):
        for r in range(ROWS - 3):
            out.append(tuple((r + k, c) for k in range(4)))
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            out.append(tuple((r + k, c + k) for k in range(4)))
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            out.append(tuple((r + k, c - k) for k in range(4)))
    return out

_WINDOWS = _iter_windows()


def _line_score_for_player(cells: Sequence[int], player: int) -> int:
    """Score one 4-cell window from *player*'s perspective (0 if contested)."""
    pieces = [cells[i] for i in range(4)]
    if player not in pieces:
        return 0
    others = [x for x in pieces if x != 0 and x != player]
    if others:
        return 0
    mine = pieces.count(player)
    empties = pieces.count(0)
    if mine + empties != 4:
        return 0
    if mine == 4:
        return 10_000
    if mine == 3 and empties == 1:
        return 120
    if mine == 2 and empties == 2:
        return 18
    if mine == 1 and empties == 3:
        return 2
    return 0


def _evaluate_board(board: Board, root: int, num_p: int) -> int:
    """Paranoid-style scalar: root strength minus strongest opponent threat."""
    per = {p: 0 for p in range(1, num_p + 1)}
    g = board.grid
    for w in _WINDOWS:
        vals = [g[r][c] for r, c in w]
        for p in range(1, num_p + 1):
            per[p] += _line_score_for_player(vals, p)
    mine = per[root]
    opp = max(per[p] for p in range(1, num_p + 1) if p != root)
    return mine - opp


# --- Random --------------------------------------------------------------------

class RandomAI:
    """Chọn cột hợp lệ ngẫu nhiên."""

    def get_move(self, board: Board, cur: int, num_p: int) -> int:
        moves = _valid_moves(board)
        if not moves:
            return -1
        return random.choice(moves)


# --- Minimax (paranoid alpha–beta) ---------------------------------------------

_WIN = 1_000_000
_LOSS = -1_000_000


class MinimaxAI:
    """
    Minimax độ sâu (2–6 nửa-nước / plies), heuristic theo cửa sổ 4 ô.
    Gốc: người *cur* tối đa hóa; các đối thủ tối thiểu hóa điểm của *cur*.
    """

    __slots__ = ("depth",)

    def __init__(self, depth: int = 4):
        self.depth = max(2, min(6, int(depth)))

    def get_move(self, board: Board, cur: int, num_p: int) -> int:
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
            w = board.check_win_at(r, col)
            if w:
                board.undo(col)
                return col
            nxt = _next_player(cur, num_p)
            v, _ = self._search(board, self.depth - 1, alpha, beta, nxt, cur, num_p, False)
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
        maximizing_for_root: bool,
    ) -> Tuple[float, Optional[int]]:
        if board.is_full():
            return 0.0, None

        if depth == 0:
            return float(_evaluate_board(board, root, num_p)), None

        moves = _valid_moves(board)
        if not moves:
            return 0.0, None

        if maximizing_for_root:
            best = -math.inf
            best_c: Optional[int] = None
            for col in moves:
                r = board.drop(col, to_move)
                if r < 0:
                    continue
                win = board.check_win_at(r, col)
                if win == root:
                    board.undo(col)
                    return _WIN, col
                if win and win != root:
                    board.undo(col)
                    return _LOSS, col
                nxt = _next_player(to_move, num_p)
                child_max = nxt == root
                v, _ = self._search(board, depth - 1, alpha, beta, nxt, root, num_p, child_max)
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
            if win == root:
                board.undo(col)
                return _LOSS, col
            if win and win != root:
                board.undo(col)
                return _LOSS, col
            nxt = _next_player(to_move, num_p)
            child_max = nxt == root
            v, _ = self._search(board, depth - 1, alpha, beta, nxt, root, num_p, child_max)
            board.undo(col)
            if v < best:
                best = v
                best_c = col
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best, best_c


# --- MCTS (UCT, đa người chơi) -------------------------------------------------

class _MCTSNode:
    __slots__ = ("parent", "move", "who", "children", "_untried", "N", "W")

    def __init__(self, parent: Optional["_MCTSNode"], move: int, who: Optional[int]):
        self.parent = parent
        self.move = move
        self.who = who
        self.children: dict[int, _MCTSNode] = {}
        self._untried: Optional[List[int]] = None
        self.N = 0
        self.W = 0.0


class MCTSAI:
    """
    Monte Carlo Tree Search với UCT; số rollout cấu hình được (200–5000).
    Mỗi nút lưu thống kê cho người vừa đi nước tới nút đó (phù hợp đa người chơi).
    """

    __slots__ = ("simulations", "explore_c")

    def __init__(self, simulations: int = 1000, explore_c: float = math.sqrt(2.0)):
        self.simulations = max(200, min(5000, int(simulations)))
        self.explore_c = explore_c

    def get_move(self, board: Board, cur: int, num_p: int) -> int:
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

        root = _MCTSNode(None, -1, None)
        root._untried = list(moves)

        for _ in range(self.simulations):
            b = board.copy()
            path: List[_MCTSNode] = []
            node = root
            to_move = cur

            while True:
                w = b.winner()
                if w != 0 or b.is_full():
                    self._backprop(path, w)
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
                    self._rollout(b, to_move, num_p)
                    self._backprop(path, b.winner())
                    break

                if not node.children:
                    self._rollout(b, to_move, num_p)
                    self._backprop(path, b.winner())
                    break

                ch = self._uct_select_child(node)
                r = b.drop(ch.move, to_move)
                if r < 0:
                    self._rollout(b, to_move, num_p)
                    self._backprop(path, b.winner())
                    break
                path.append(ch)
                to_move = _next_player(to_move, num_p)
                node = ch

        if not root.children:
            return random.choice(moves)
        best_col = max(
            root.children.keys(),
            key=lambda c: (root.children[c].N, random.random()),
        )
        return best_col

    def _uct_select_child(self, node: _MCTSNode) -> _MCTSNode:
        children = list(node.children.values())
        total_n = sum(ch.N for ch in children)
        log_n = math.log(total_n + 1e-9)
        best: List[_MCTSNode] = []
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
                best = [ch]
            elif math.isclose(ucb, best_s) or (ucb == math.inf and best_s == math.inf):
                best.append(ch)
        return random.choice(best)

    def _rollout(self, b: Board, to_move: int, num_p: int) -> None:
        p = to_move
        while not b.is_full():
            opts = _valid_moves(b)
            if not opts:
                break
            col = random.choice(opts)
            r = b.drop(col, p)
            if r < 0:
                break
            if b.check_win_at(r, col):
                break
            p = _next_player(p, num_p)

    def _backprop(self, path: List[_MCTSNode], winner: int) -> None:
        for node in path:
            node.N += 1
            if winner != 0 and node.who is not None and node.who == winner:
                node.W += 1.0

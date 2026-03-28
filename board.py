"""Grid logic, O(1) drops, and win detection for 8×7 three-player Connect Four."""

import random as _rng
from constants import ROWS, COLS

_zrng = _rng.Random(314159)
_ZOBRIST = [[[_zrng.getrandbits(64) for _ in range(4)]
             for _ in range(COLS)] for _ in range(ROWS)]


class Board:
    """Game grid, move validation, win checks, and Zobrist hash for transpositions."""

    __slots__ = ['grid', '_heights', '_hash', '_move_count']

    _DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))

    def __init__(self):
        self.grid = [[0] * COLS for _ in range(ROWS)]
        self._heights = [ROWS - 1] * COLS
        self._hash = 0
        self._move_count = 0

    def copy(self):
        """Independent copy of the board state."""
        b = Board.__new__(Board)
        b.grid = [row[:] for row in self.grid]
        b._heights = self._heights[:]
        b._hash = self._hash
        b._move_count = self._move_count
        return b

    def is_valid(self, col):
        """True if *col* can accept a piece."""
        return 0 <= col < COLS and self._heights[col] >= 0

    def valid_moves(self):
        """List of columns that are not full."""
        return [c for c in range(COLS) if self._heights[c] >= 0]

    def drop_row(self, col):
        """Row index where a piece would land, or -1 if column full."""
        return self._heights[col]

    def drop(self, col, player):
        """Place *player* in *col*; return landing row or -1 if invalid."""
        r = self._heights[col]
        if r >= 0:
            self.grid[r][col] = player
            self._heights[col] = r - 1
            self._hash ^= _ZOBRIST[r][col][player]
            self._move_count += 1
        return r

    def undo(self, col):
        """Remove top piece from *col* (inverse of drop)."""
        r = self._heights[col] + 1
        if r < ROWS:
            p = self.grid[r][col]
            self._hash ^= _ZOBRIST[r][col][p]
            self.grid[r][col] = 0
            self._heights[col] = r
            self._move_count -= 1

    def check_win_at(self, r, c):
        """Winning player if piece at (r,c) completes four in a row, else 0."""
        p = self.grid[r][c]
        if p == 0:
            return 0
        g = self.grid
        for dr, dc in Board._DIRS:
            count = 1
            nr, nc = r + dr, c + dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and g[nr][nc] == p:
                count += 1
                if count >= 4:
                    return p
                nr += dr
                nc += dc
            nr, nc = r - dr, c - dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and g[nr][nc] == p:
                count += 1
                if count >= 4:
                    return p
                nr -= dr
                nc -= dc
        return 0

    def winning_cells_at(self, r, c):
        """Four cells forming a win through (r,c), or empty list."""
        p = self.grid[r][c]
        if p == 0:
            return []
        g = self.grid
        for dr, dc in Board._DIRS:
            cells = [(r, c)]
            nr, nc = r + dr, c + dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and g[nr][nc] == p:
                cells.append((nr, nc))
                nr += dr
                nc += dc
            nr, nc = r - dr, c - dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and g[nr][nc] == p:
                cells.append((nr, nc))
                nr -= dr
                nc -= dc
            if len(cells) >= 4:
                return cells[:4]
        return []

    def winner(self):
        """Any winning player on the full board (O(rows×cols)); prefer check_win_at after moves."""
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p and self._line(r, c, p):
                    return p
        return 0

    def _line(self, r, c, p):
        for dr, dc in self._DIRS:
            ok = True
            for i in range(1, 4):
                nr, nc = r + dr * i, c + dc * i
                if not (0 <= nr < ROWS and 0 <= nc < COLS) or self.grid[nr][nc] != p:
                    ok = False
                    break
            if ok:
                return True
        return False

    def winning_cells(self):
        """Coordinates of one winning line of four, or empty."""
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if not p:
                    continue
                for dr, dc in self._DIRS:
                    cells = [(r + dr * i, c + dc * i) for i in range(4)]
                    if all(
                        0 <= cr < ROWS and 0 <= cc < COLS and self.grid[cr][cc] == p
                        for cr, cc in cells
                    ):
                        return cells
        return []

    def is_full(self):
        """No empty cells remain."""
        return self._move_count >= ROWS * COLS

    def is_terminal(self):
        """Win or draw."""
        return self.winner() != 0 or self.is_full()

    def to_list(self):
        """Grid as list of rows (for JSON)."""
        return [row[:] for row in self.grid]

    @classmethod
    def from_list(cls, data):
        """Rebuild board, heights, hash, and move count from saved rows."""
        b = cls()
        b.grid = [row[:] for row in data]
        b._move_count = 0
        b._hash = 0
        for c in range(COLS):
            b._heights[c] = -1
            for r in range(ROWS):
                if data[r][c] != 0:
                    b._move_count += 1
                    b._hash ^= _ZOBRIST[r][c][data[r][c]]
            for r in range(ROWS - 1, -1, -1):
                if data[r][c] == 0:
                    b._heights[c] = r
                    break
        return b

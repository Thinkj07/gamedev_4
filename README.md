# Connect Four · 3 Players

A polished, turn-based board game for **2–3 players** built entirely with
**Python 3** and **Pygame**.  Three coloured discs (Red, Yellow, Blue) compete
on an **8 × 7** grid – the first to connect four in a row wins!

---

## Requirements

| Dependency | Version |
|------------|---------|
| Python     | 3.9 +   |
| Pygame     | 2.0 +   |

## Installation

```bash
pip install pygame
```

## Running the game

```bash
python main.py
```

---

## Game Modes

### SinglePlayer
Play alone against AI opponents. Choose to face **1 or 2 AI bots**.
- 1 AI → 2-player game (You vs AI)
- 2 AI → 3-player game (You vs AI vs AI)

### Multiplayer
Play with friends on the same device. Add **1 or 2 extra players**.
- +1 Player → 2-player game
- +2 Players → 3-player game

## Game rules

1. Players take turns dropping a disc into one of the 8 columns.
2. The disc falls to the lowest empty cell in that column.
3. The first player to align **4 discs in a row** (horizontally, vertically,
   or diagonally) **wins**.
4. If the board fills up with no winner, the game is a **draw**.

## AI opponents

In SinglePlayer mode, AI slots can be configured with one of three algorithms:

| AI | Description |
|----|-------------|
| **Random** | Picks a legal column uniformly at random. |
| **MCTS** | Monte-Carlo Tree Search (UCT). Configurable simulation count (200 – 5 000). |
| **Minimax** | Multiplayer Max^n search. Configurable depth (2 – 6). Includes a heuristic that evaluates potential lines and centre control. |

### Configuration

After selecting a game mode and player count, configure AI type and difficulty
by clicking on the selector buttons in the config screen.

## Controls

| Action | Control |
|--------|---------|
| Drop a disc | Left-click on a column |
| Pause | `Esc` key |
| Resume | `Esc` key or *Resume* button |
| Save & Quit | Pause menu → *Save & Quit* |
| Continue saved game | Main menu → *Continue* |

## Save / Load

The game state (board, current player, AI settings, game mode) is saved to
`savegame.json` when you choose **Save & Quit**.  Select **Continue** from the
main menu to resume exactly where you left off.

## UI Design

The interface uses a **60-30-10** color rule:
- **60% Cream/Off-white** – background and panels
- **30% Dark Teal** – board, buttons, and accents
- **10% Player Colors** – Red, Yellow, Blue for disc highlights

## Project structure

```
boardgame_3/
├── main.py          # Entry point
├── app.py           # Application: menus, rendering, game loop
├── board.py         # Board logic and win detection
├── ai.py            # AI algorithms (Random, MCTS, Minimax)
├── constants.py     # All game constants and colours
├── savegame.json    # Auto-generated save file
└── README.md        # This file
```

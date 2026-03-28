# Connect Four · 3 Players

A polished, turn-based board game for **2–3 players** built entirely with
**Python 3** and **Pygame**.  Three coloured discs (Red, Yellow, Blue) compete
on an **8 × 7** grid – the first to connect four in a row wins!

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.9 + | Runtime |
| Pygame | 2.0 + | Game rendering |
| scikit-learn | 1.3 + | ML model training & inference |
| pandas | 1.5 + | Data loading in trainer |
| joblib | 1.2 + | Model serialisation |
| numpy | 1.23 + | Feature vectors |

## Installation

```bash
pip install pygame scikit-learn pandas joblib numpy
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

---

## Game Rules

1. Players take turns dropping a disc into one of the 8 columns.
2. The disc falls to the lowest empty cell in that column.
3. The first player to align **4 discs in a row** (horizontally, vertically,
   or diagonally) **wins**.
4. If the board fills up with no winner, the game is a **draw**.

---

## AI Opponents

In SinglePlayer mode, AI slots can be configured with one of five algorithms:

| AI | Description |
|----|-------------|
| **Random** | Picks a legal column uniformly at random. |
| **MCTS** | Monte-Carlo Tree Search (UCT). Configurable simulation count (200 – 5 000). Uses random rollouts. |
| **Minimax** | Paranoid alpha–beta search. Configurable depth (2 – 6). Uses a hand-crafted window heuristic. |
| **ML-Minimax** | Same paranoid alpha–beta tree as Minimax, but **leaf evaluation** is performed by a trained `MLPClassifier` instead of the hand-crafted heuristic. |
| **ML-MCTS** | UCT MCTS where the expensive random rollout is **replaced** by a single ML model inference at each leaf expansion. Substantially faster per simulation than standard MCTS. |

### ML Paranoid Score

Both ML AI types share the same evaluation formula:

```
score = P(root_player wins) − max(P(opponent_i wins))   ∈ [−1, 1]
```

where the probabilities come from `model.predict_proba()` on the current board state.

> **Note:** If `connect4_ml_model.pkl` is not found, `ML-Minimax` and `ML-MCTS`
> fall back gracefully to the hand-crafted heuristic / random rollouts and print
> a warning. Run the ML pipeline (below) to generate the model.

### Configuration

After selecting a game mode and player count, configure AI type and difficulty
by clicking the selector buttons in the config screen.

---

## ML Pipeline

The ML system is split into three standalone scripts that must be run in order.

### Phase 1 — Data Generation

Simulates self-play games using `RandomAI` for all players (randomly 2-player
and 3-player) and records every intermediate board state alongside the final
winner.

```bash
python data_generator.py                 # 5 000 games (default)
python data_generator.py --games 20000   # more data = better model
python data_generator.py --seed 42       # reproducible run
```

**Output:** `connect4_data.csv`
- 56 cell columns (`cell_0` … `cell_55`, row-major 7×8 grid)
- `num_p` — number of players in that game (2 or 3)
- `winner` — 0 (draw), 1, 2, or 3

### Phase 2 — Model Training

Loads the CSV, trains an `MLPClassifier` (256 → 128 → 64 hidden units, Adam,
early stopping), and serialises the fitted pipeline to disk.

```bash
python train_model.py                        # defaults
python train_model.py --data my_data.csv     # custom input
python train_model.py --epochs 500 --lr 5e-4 # tuning
```

**Output:** `connect4_ml_model.pkl` — a scikit-learn `Pipeline`
(StandardScaler + MLPClassifier).  
Prints test-set accuracy and a full classification report.

### Phase 3 — Play with ML AI

Once the model file exists, simply start the game and select **ML-Minimax** or
**ML-MCTS** from the config screen.

```bash
python main.py
```

### Full Pipeline (one shot)

```bash
python data_generator.py --games 5000 --seed 42
python train_model.py
python main.py
```

---

## Controls

| Action | Control |
|--------|---------|
| Drop a disc | Left-click on a column |
| Pause | `Esc` key |
| Resume | `Esc` key or *Resume* button |
| Save & Quit | Pause menu → *Save & Quit* |
| Continue saved game | Main menu → *Continue* |

---

## Save / Load

The game state (board, current player, AI settings, game mode) is saved to
`savegame.json` when you choose **Save & Quit**.  Select **Continue** from the
main menu to resume exactly where you left off.

---

## UI Design

The interface uses a **60-30-10** colour rule:
- **60% Cream/Off-white** — background and panels
- **30% Dark Teal** — board, buttons, and accents
- **10% Player Colours** — Red, Yellow, Blue for disc highlights

---

## Project Structure

```
boardgame_3/
├── main.py               # Entry point
├── app.py                # Application: menus, rendering, game loop
├── board.py              # Board logic and win detection
├── constants.py          # All game constants and colours
├── ai.py                 # Classic AI: Random, MCTS, Minimax
├── ai_ml.py              # ML AI: ML_MinimaxAI, ML_MCTSAI
├── data_generator.py     # Phase 1 – self-play data collection
├── train_model.py        # Phase 2 – MLP classifier training
├── connect4_data.csv     # Generated training data  (git-ignored)
├── connect4_ml_model.pkl # Trained model            (git-ignored)
├── savegame.json         # Auto-generated save file
└── README.md             # This file
```

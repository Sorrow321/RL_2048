# RL 2048 — AlphaZero with a C++ MCTS engine

<p align="center">
  <img src="assets/demo.gif" width="320"
       alt="Trained AlphaZero agent playing 2048: 4,676 moves, score 112,916, reaching the 8192 tile">
  <br>
  <em>The trained agent reaching the 8192 tile — 4,676 moves, score 112,916
  (10× sped up; render your own with <code>az.render_gif</code>)</em>
</p>

AlphaZero for 2048: the game and the search trees live in C++ (bitboard
engine, PUCT MCTS, sampled chance nodes), PyTorch stays in Python. The
bridge is batched — a `Runner` steps hundreds of concurrent games, pauses
each at leaves needing evaluation, and Python answers all of them with a
single GPU forward per cycle.

## Results

Trained overnight on a single RTX 5090 (~7h wall clock, ~60 generations of
256 self-play games): the 840k-parameter net reaches **8192** in ~1–2% of
games and **4096** in ~44% when evaluated at 1536 sims/move, with mean
score ~47k. The net surpasses the heuristic teacher that bootstrapped it
by generation 14, using 40× fewer simulations per move.

## Layout

- `cpp/az_engine.cpp` — pybind11 module: batched AlphaZero MCTS runner
  plus small game utilities. Action ids: 0=down 1=up 2=right 3=left.
- `pure_cpp/run_2048_mcts.cpp` — standalone heuristic MCTS
  (no NN, ~3000 moves/s root-parallel). Two modes:
  - play one game: `./run_2048_mcts` (env: `MCTS_ITERS`, `MCTS_THREADS`)
  - self-play dump for bootstrapping: `MCTS_DUMP=prefix MCTS_GAMES=400
    MCTS_ITERS=10000 ./run_2048_mcts`
- `az/` — Python package: net, data, self-play driver, training, eval.

## Build

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(which python) -DPYBIND11_FINDPYTHON=ON
cmake --build build -j            # -> ./az_engine*.so at repo root
g++ -std=c++17 -O3 -march=native -pthread pure_cpp/run_2048_mcts.cpp \
    -o pure_cpp/run_2048_mcts
```

## Train

```bash
# 1) bootstrap an initial net from heuristic-MCTS self-play (~2 min total)
python -m az.bootstrap --out runs/bootstrap

# 2) AlphaZero loop: self-play -> replay buffer -> train -> eval, per generation
python -m az.train --run runs/az0 --bootstrap-net runs/bootstrap/net.pt

# 3) evaluate any checkpoint
python -m az.eval --ckpt runs/az0/latest.pt --games 32 --sims 128
python -m az.eval --ckpt runs/az0/latest.pt --games 100 --greedy
```

Metrics per generation go to `<run>/log.csv`.

## Design notes

- Value target: `log1p(future merge score) / 12` — death is exactly 0,
  strong late-game positions approach ~1. The tree backs up leaf values
  only; terminal leaves back up 0 without calling the net.
- Policy target: PUCT root visit distributions. (The bootstrap trains on
  hard argmax labels instead — UCB1 visit counts in the heuristic search
  are near-uniform and carry little signal.)
- Chance nodes sample spawns from the true 90/10 distribution with
  progressive widening and deduplication.
- Trees are arena-allocated and rebuilt each move: with 2048's chance
  branching, a reused subtree retains only ~1% of visits (measured).

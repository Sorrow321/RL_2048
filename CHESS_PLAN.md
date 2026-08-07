# Chess AlphaZero — plan

Goal: a chess agent that clearly clears ~1000 Elo after one overnight run,
with a path to 1800+ over subsequent nights. Reuses the architecture proven
in this repo on 2048 (batched C++ MCTS bridge + PyTorch in Python).
Hardware baseline: RTX 5090 + 16 cores.

## What carries over from the 2048 pipeline

| Piece | Status |
|---|---|
| Batched bridge design (`cpp/az_engine.cpp` Runner: C++ trees pause at leaves, one GPU batch per cycle) | Reuse structure as-is |
| Generation loop (`az/train.py`: self-play → buffer → train → eval, CSV logging, `--target` stop) | Reuse with new metrics |
| Bootstrap-from-expert pattern (`az/bootstrap.py`) | Same idea, new expert (human games) |
| Arena-allocated trees, PUCT, Dirichlet root noise | Reuse |
| Plateau playbook (raise sims → lr decay → data reweighting) and dashboard renderer | Reuse |

Key differences to respect: sparse terminal reward (win/draw/loss, no dense
score), ~4672-way policy, much bigger net, and no chance nodes (chess is
deterministic — the tree gets simpler: pure two-player minimax-style PUCT
with value negation between plies).

## Phase 0 — Engine integration (the real cost: ~1–2 days)

- **Do not write chess movegen.** Embed an existing C++ library:
  first candidate [Disservin/chess-library](https://github.com/Disservin/chess-library)
  (header-only, fast, clean API); alternative: surge, or extracting Stockfish movegen.
- Validate with **perft tests** (standard positions, depths 5–6, exact node
  counts) before anything else. Movegen bugs poison everything downstream.
- Implement game-end handling: checkmate, stalemate, 50-move, threefold
  repetition (needs position hashing), insufficient material. Draw
  adjudication for self-play (cap at ~250 plies, score as draw).
- Board encoding → planes (AlphaZero-style): 12 piece planes + side to move,
  castling rights ×4, en-passant file, halfmove clock. History planes
  optional at this scale — start without, note as lever.
- Move encoding: 8×8×73 policy planes (queen moves ×56, knight ×8,
  underpromotions ×9) = 4672 logits, legal-masked. Write both directions
  (move→index, index→move) with round-trip tests over perft-generated moves.

## Phase 1 — Supervised bootstrap (one overnight, near-guarantees the goal)

The Maia result: a policy-only net trained on human Lichess games plays at
the level of its training data (their 1100–1900 models are exactly this).

- Download Lichess database PGNs (a few months, filter ~1200–2000 rated
  blitz/rapid; tens of millions of positions).
- Targets: policy = move played (hard label — same lesson as the 2048
  bootstrap), value = game outcome from mover's perspective {-1, 0, +1}.
- Net: ResNet, 6 blocks × 128 filters (~5M params) to start. Policy head
  4672, value head tanh scalar.
- Sanity gate (analog of greedy-2048 eval): raw policy argmax vs
  Stockfish skill-1 and vs random. Expect >1000 Elo behavior from the
  policy alone if the data pipeline is right.

## Phase 2 — Eval harness (build BEFORE self-play; ~half a day)

Elo claims need opponents. Script matches vs Stockfish at fixed skill
levels / node limits (python-chess UCI is fine here, it's not the hot path),
~100 games per match point, compute Elo from score with error bars.
This is the chess analog of the 2048 dashboard's eval column — wire it into
the generation loop as the per-generation eval.

## Phase 3 — AlphaZero self-play loop (nights 2+)

- Port the Runner: deterministic tree (no chance nodes), value negated per
  ply, terminal values from game result (+1/0/-1), Dirichlet α≈0.3 at root,
  temperature=1 move sampling for the first ~15 plies then argmax
  (chess needs opening diversity; 2048 got it free from spawns).
- Throughput estimate: ~120 plies × 400 sims, 256 games in flight,
  5M-param net → ~5–8k games/hour → 40–60k games/night.
- Replay buffer ~1M positions; sample late-game and decisive-game positions
  at a modest premium (draw-heavy buffers stall the value head).
- Expected trajectory (hobby-scale reference points): bootstrap+search
  ~1100–1400 → few nights of self-play ~1600–1900 → diminishing returns.
  Superhuman is not the goal; it is a different compute regime.

## Known risks / lessons imported from 2048

- Hard labels for bootstrap policy (soft targets from weak search were
  near-uniform noise in 2048; human moves are already hard labels).
- Sims per move gate target quality — when progress plateaus, raise
  self-play sims before touching anything else.
- Search amplification at eval is the cheap win: evaluate at 2–4× the
  self-play sims when chasing a milestone.
- Verify every low-level component against a reference implementation
  before trusting it (the 2048 engine equivalence tests caught real bugs;
  perft + encoding round-trips play that role here).
- Watch for orphaned training processes when restarting runs (kill the
  python PID, verify with pgrep).

## Open decisions (fine to defer)

- Movegen library final pick (benchmark 2–3 candidates on perft speed).
- History planes in the encoding (AZ used 8 half-moves; start with 0–1).
- Net size schedule (start 6×128, widen when self-play plateaus).
- Whether to shard self-play across the 2 CUDA MPS contexts or keep one
  process (start simple: one process, batched, like 2048).

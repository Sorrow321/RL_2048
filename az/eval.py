"""Evaluate a checkpoint: MCTS play (noise-free) or raw-policy greedy play.

    python -m az.eval --ckpt runs/az0/latest.pt --games 32 --sims 128
    python -m az.eval --ckpt runs/az0/latest.pt --games 100 --greedy
"""
import argparse

import numpy as np
import torch

import az_engine
from az.net import load_checkpoint, encode_boards
from az.selfplay import play_games, result_stats


def play_greedy(net, device, n_games, seed=0):
    """All games stepped in lockstep; one forward per step for the batch."""
    boards, rngs = [], []
    for i in range(n_games):
        b, r = az_engine.new_game(seed * 1_000_003 + i + 1)
        boards.append(b)
        rngs.append(r)
    scores = [0] * n_games
    active = list(range(n_games))
    results = []

    with torch.inference_mode():
        while active:
            batch = np.array([boards[i] for i in active], dtype=np.uint64)
            logits, _ = net(encode_boards(batch, device))
            logits = logits.cpu().numpy()
            still = []
            for row, i in enumerate(active):
                legal = az_engine.legal_actions(boards[i])
                if not legal:
                    results.append((scores[i], az_engine.max_tile(boards[i]), 0))
                    continue
                mask = np.full(4, -np.inf)
                mask[legal] = 0.0
                a = int((logits[row] + mask).argmax())
                nb, sc, _ = az_engine.do_move(boards[i], a)
                scores[i] += sc
                boards[i], rngs[i] = az_engine.spawn(nb, rngs[i])
                if az_engine.can_move(boards[i]):
                    still.append(i)
                else:
                    results.append((scores[i], az_engine.max_tile(boards[i]), 0))
            active = still
    return np.array(results, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--greedy", action="store_true", help="raw policy, no search")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_checkpoint(args.ckpt, device)
    net.eval()

    if args.greedy:
        res = play_greedy(net, device, args.games, args.seed)
    else:
        _, res = play_games(net, device, args.games, args.sims,
                            dirichlet_eps=0.0, seed=args.seed)

    stats = result_stats(res)
    mode = "greedy" if args.greedy else f"MCTS sims={args.sims}"
    print(f"{mode}, {stats['games']} games: "
          f"mean score {stats['score_mean']:.0f} (max {stats['score_max']})")
    for t, c in sorted(stats["tiles"].items()):
        print(f"  max tile {t}: {c}/{stats['games']}")


if __name__ == "__main__":
    main()

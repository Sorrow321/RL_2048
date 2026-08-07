"""Watch the trained agent play.

Record: play N games with MCTS (no noise), save the best game's full
trajectory (boards, chosen moves, root visit counts, rewards) as JSON:

    python -m az.watch --ckpt runs/hunt2/latest.pt --games 8 --sims 256 \
        --out runs/best_game.json

Replay in the terminal:

    python -m az.watch --show runs/best_game.json --fps 15
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

import az_engine
from az.net import load_checkpoint
from az.selfplay import play_games

ARROWS = {0: "down", 1: "up", 2: "right", 3: "left"}


def record(ckpt, games, sims, out, seed, device):
    net = load_checkpoint(ckpt, device)
    net.eval()
    runner_examples, results = play_games(
        net, device, games, sims, n_parallel=games,
        dirichlet_eps=0.0, seed=seed)

    # split flat examples into games, pick the best by total reward
    done = runner_examples["done"]
    bounds = np.concatenate([[0], np.flatnonzero(done == 1) + 1])
    best, best_score = None, -1
    for s, e in zip(bounds[:-1], bounds[1:]):
        score = int(runner_examples["reward"][s:e].sum())
        if score > best_score:
            best_score, best = score, (s, e)

    s, e = best
    moves = []
    for i in range(s, e):
        moves.append({
            "board": f"{int(runner_examples['board'][i]):016x}",
            "action": int(runner_examples["chosen"][i]),
            "visits": [int(v) for v in runner_examples["visits"][i]],
            "reward": int(runner_examples["reward"][i]),
        })
    # final position: apply the last move (pre-spawn board is accurate
    # up to the final 2/4 spawn, which never changes the max tile)
    last_board, _, _ = az_engine.do_move(
        int(runner_examples["board"][e - 1]),
        int(runner_examples["chosen"][e - 1]))
    game = {
        "score": best_score,
        "max_tile": int(az_engine.max_tile(last_board)),
        "steps": int(e - s),
        "sims": sims,
        "final_board": f"{last_board:016x}",
        "moves": moves,
    }
    with open(out, "w") as f:
        json.dump(game, f)
    print(f"recorded best of {games} games -> {out}")
    print(f"score {best_score}, max tile {game['max_tile']}, {game['steps']} moves")


def board_lines(board_hex):
    b = int(board_hex, 16)
    lines = []
    for r in range(4):
        row = []
        for c in range(4):
            v = (b >> (4 * (r * 4 + c))) & 0xF
            row.append(f"{(1 << v) if v else '.':>6}")
        lines.append(" ".join(row))
    return lines


def show(path, fps):
    game = json.load(open(path))
    total = 0
    for i, m in enumerate(game["moves"]):
        total += m["reward"]
        sys.stdout.write("\x1b[2J\x1b[H")  # clear screen
        print(f"move {i + 1}/{game['steps']}   score {total}   "
              f"-> {ARROWS[m['action']]}\n")
        print("\n".join(board_lines(m["board"])))
        pi = np.array(m["visits"], dtype=float)
        pi /= max(pi.sum(), 1)
        print("\n" + "  ".join(
            f"{ARROWS[a]} {pi[a]:.0%}" for a in range(4)))
        time.sleep(1.0 / fps)
    print(f"\ngame over: score {game['score']}, max tile {game['max_tile']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--out", default="runs/best_game.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", help="replay a recorded JSON in the terminal")
    ap.add_argument("--fps", type=float, default=15)
    args = ap.parse_args()

    if args.show:
        show(args.show, args.fps)
    else:
        if not args.ckpt:
            ap.error("--ckpt required for recording")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        record(args.ckpt, args.games, args.sims, args.out, args.seed, device)


if __name__ == "__main__":
    main()

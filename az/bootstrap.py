"""Bootstrap: generate heuristic-MCTS self-play data with the pure C++
binary (fast: no NN in the loop), then train an initial policy+value net.

    python -m az.bootstrap --out runs/bootstrap --games 400 --iters 10000

Policy loss uses HARD labels on the searched action: UCB1 root visit
distributions are near-uniform (measured entropy 1.20 vs ln4=1.386), so
soft targets carry almost no signal — unlike the PUCT visits used later
in az.train.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from az.net import PolicyValueNet, encode_boards, save_checkpoint
from az.data import load_dump, make_targets

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO, "pure_cpp", "run_2048_mcts")


def ensure_binary():
    src = os.path.join(REPO, "pure_cpp", "run_2048_mcts.cpp")
    if os.path.exists(BINARY) and os.path.getmtime(BINARY) >= os.path.getmtime(src):
        return
    print("compiling pure_cpp binary...")
    subprocess.run(
        ["g++", "-std=c++17", "-O3", "-march=native", "-pthread", src, "-o", BINARY],
        check=True)


def generate(prefix, games, iters):
    ensure_binary()
    env = dict(os.environ, MCTS_DUMP=prefix, MCTS_GAMES=str(games),
               MCTS_ITERS=str(iters))
    subprocess.run([BINARY], env=env, check=True)


def train(prefix, out_path, epochs, device):
    d = load_dump(prefix)
    boards, _, z = make_targets(d)
    chosen = torch.from_numpy(d["chosen"].astype(np.int64))
    n = len(boards)
    n_val = max(1, n // 10)
    perm0 = np.random.permutation(n)  # dump files interleave threads; shuffle
    tr, va = perm0[:-n_val], perm0[-n_val:]
    print(f"{n} positions (train {len(tr)}, val {len(va)})")

    z_t = torch.from_numpy(z)
    net = PolicyValueNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    B = 4096

    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr)
        p_sum = v_sum = nb = 0
        for i in range(0, len(perm), B):
            idx = perm[i:i + B]
            logits, v = net(encode_boards(boards[idx], device))
            lp = F.cross_entropy(logits, chosen[idx].to(device))
            lv = F.mse_loss(v, z_t[idx].to(device))
            opt.zero_grad()
            (lp + lv).backward()
            opt.step()
            p_sum += lp.item(); v_sum += lv.item(); nb += 1

        net.eval()
        correct = tot = 0
        with torch.inference_mode():
            for i in range(0, len(va), B):
                idx = va[i:i + B]
                logits, _ = net(encode_boards(boards[idx], device))
                correct += (logits.argmax(1).cpu() == chosen[idx]).sum().item()
                tot += len(idx)
        print(f"epoch {ep}: p={p_sum/nb:.4f} v={v_sum/nb:.5f} "
              f"val-agree={correct/tot:.1%}")

    save_checkpoint(net, out_path)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--skip-generate", action="store_true",
                    help="reuse an existing dump in --out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.join(args.out, "dump")
    if not args.skip_generate:
        generate(prefix, args.games, args.iters)
    train(prefix, os.path.join(args.out, "net.pt"), args.epochs, device)


if __name__ == "__main__":
    main()

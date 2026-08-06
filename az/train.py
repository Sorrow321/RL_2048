"""AlphaZero training loop for 2048.

    python -m az.train --run runs/az0 --bootstrap-net runs/bootstrap/net.pt \
        --generations 30

Each generation: self-play with the current net (root Dirichlet noise on),
push examples into the replay buffer, take gradient steps on sampled
batches, checkpoint, and evaluate with noise-free MCTS play. Metrics go to
<run>/log.csv.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from az.net import PolicyValueNet, encode_boards, save_checkpoint, load_checkpoint
from az.data import ReplayBuffer, make_targets
from az.selfplay import play_games, result_stats


def train_steps(net, opt, buffer, n_steps, batch_size, device, rng):
    """Gradient steps on sampled buffer batches. pi may be soft or one-hot."""
    net.train()
    p_sum = v_sum = 0.0
    for _ in range(n_steps):
        boards, pi, z = buffer.sample(batch_size, rng)
        x = encode_boards(boards, device)
        pi_t = torch.from_numpy(pi).to(device)
        z_t = torch.from_numpy(z).to(device)
        logits, v = net(x)
        loss_p = -(pi_t * F.log_softmax(logits, dim=1)).sum(1).mean()
        loss_v = F.mse_loss(v, z_t)
        opt.zero_grad()
        (loss_p + loss_v).backward()
        opt.step()
        p_sum += loss_p.item()
        v_sum += loss_v.item()
    return p_sum / n_steps, v_sum / n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir (checkpoints, log.csv)")
    ap.add_argument("--bootstrap-net", default=None,
                    help="initial checkpoint (from az.bootstrap); random init if omitted")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--games-per-gen", type=int, default=256)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--n-parallel", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--steps-per-gen", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.5)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--eval-games", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.run, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.bootstrap_net:
        net = load_checkpoint(args.bootstrap_net, device)
        print(f"initialized from {args.bootstrap_net}")
    else:
        net = PolicyValueNet().to(device)
        print("random initialization (consider az.bootstrap first)")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)

    log_path = os.path.join(args.run, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["gen", "sp_score_mean", "sp_score_max", "sp_steps_mean",
                      "pi_entropy", "buffer", "loss_p", "loss_v",
                      "eval_score_mean", "eval_score_max", "eval_tiles",
                      "sp_seconds", "train_seconds", "eval_seconds"])

    for gen in range(args.generations):
        t0 = time.time()
        ex, res = play_games(
            net, device, args.games_per_gen, args.sims, args.n_parallel,
            args.c_puct, args.dirichlet_alpha, args.dirichlet_eps,
            seed=int(rng.integers(2**63)))
        sp_time = time.time() - t0
        sp = result_stats(res)

        boards, pi, z = make_targets(ex)
        ent = float(-(np.where(pi > 0, pi * np.log(pi), 0)).sum(1).mean())
        buffer.add(boards, pi, z)

        t0 = time.time()
        loss_p, loss_v = train_steps(net, opt, buffer, args.steps_per_gen,
                                     args.batch, device, rng)
        tr_time = time.time() - t0

        t0 = time.time()
        _, eres = play_games(net, device, args.eval_games, args.sims,
                             args.n_parallel, args.c_puct,
                             dirichlet_alpha=args.dirichlet_alpha, dirichlet_eps=0.0,
                             seed=int(rng.integers(2**63)))
        ev_time = time.time() - t0
        ev = result_stats(eres)

        save_checkpoint(net, os.path.join(args.run, f"gen_{gen:03d}.pt"))
        save_checkpoint(net, os.path.join(args.run, "latest.pt"))
        log.writerow([gen, f"{sp['score_mean']:.0f}", sp["score_max"],
                      f"{sp['steps_mean']:.0f}", f"{ent:.3f}", len(buffer),
                      f"{loss_p:.4f}", f"{loss_v:.5f}",
                      f"{ev['score_mean']:.0f}", ev["score_max"], str(ev["tiles"]),
                      f"{sp_time:.0f}", f"{tr_time:.0f}", f"{ev_time:.0f}"])
        log_f.flush()
        print(f"gen {gen}: selfplay {sp['score_mean']:.0f} "
              f"(max {sp['score_max']}, {sp_time:.0f}s, entropy {ent:.3f}) | "
              f"train p={loss_p:.4f} v={loss_v:.5f} ({tr_time:.0f}s) | "
              f"eval {ev['score_mean']:.0f} tiles {ev['tiles']} ({ev_time:.0f}s)")

    log_f.close()


if __name__ == "__main__":
    main()

"""Batched self-play: az_engine Runner fleet + one GPU forward per cycle."""
import numpy as np
import torch

import az_engine
from az.net import encode_boards


def play_games(net, device, total_games, sims, n_parallel=256,
               c_puct=1.5, dirichlet_alpha=0.5, dirichlet_eps=0.25, seed=0):
    """Play `total_games` with MCTS guided by `net`.

    dirichlet_eps=0 disables root noise (evaluation play).
    Returns (examples dict, results (N,3) int64 [score, max_tile, steps]).
    """
    runner = az_engine.Runner(
        total_games=total_games,
        n_parallel=min(n_parallel, total_games),
        sims=sims,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        seed=seed,
    )
    net.eval()
    with torch.inference_mode():
        while True:
            boards = runner.pending()
            if boards.size == 0:
                break
            logits, v = net(encode_boards(boards, device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            # tree assumes value >= 0 (unvisited Q is 0, death is 0)
            vv = v.clamp(0.0, 2.0).cpu().numpy()
            runner.feed(p, vv)
    return runner.get_examples(), runner.get_results()


def result_stats(results):
    score, tile, steps = results[:, 0], results[:, 1], results[:, 2]
    tiles = {int(t): int((tile == t).sum()) for t in np.unique(tile)}
    return {
        "games": len(results),
        "score_mean": float(score.mean()),
        "score_max": int(score.max()),
        "steps_mean": float(steps.mean()),
        "tiles": tiles,
    }

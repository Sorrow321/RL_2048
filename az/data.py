"""Self-play data handling: bootstrap dump loading, value targets, replay buffer.

Two data sources share the same fields (board, visits, reward, chosen, done):
  - the heuristic-MCTS dump written by pure_cpp/run_2048_mcts (MCTS_DUMP mode),
    32-byte records in <prefix>.tNN.bin files;
  - dicts returned by az_engine.Runner.get_examples().

The value target z is log1p(future merge score from this position) / Z_SCALE,
computed from the per-move rewards; games must be contiguous with done=1 on
their final record (both sources guarantee this).
"""
import glob
import numpy as np

from az.net import Z_SCALE

DUMP_DTYPE = np.dtype([
    ("board", "<u8"),
    ("visits", "<u4", (4,)),
    ("reward", "<u4"),
    ("chosen", "u1"),
    ("done", "u1"),
    ("pad", "<u2"),
])
assert DUMP_DTYPE.itemsize == 32


def load_dump(prefix):
    """Load <prefix>.tNN.bin files -> dict of flat arrays (games contiguous)."""
    files = sorted(glob.glob(f"{prefix}.t*.bin"))
    if not files:
        raise FileNotFoundError(f"no files match {prefix}.t*.bin")
    parts = []
    for path in files:
        recs = np.fromfile(path, dtype=DUMP_DTYPE)
        if len(recs) == 0:
            continue
        assert recs["done"][-1] == 1, f"{path}: truncated final game"
        parts.append(recs)
    recs = np.concatenate(parts)
    return {
        "board": recs["board"].copy(),
        "visits": recs["visits"].astype(np.int64),
        "reward": recs["reward"].astype(np.int64),
        "chosen": recs["chosen"].copy(),
        "done": recs["done"].copy(),
    }


def future_scores(reward, done):
    """Per-position sum of rewards to the end of its game (reversed cumsum)."""
    r = reward.astype(np.float64)
    z = np.empty_like(r)
    start = 0
    for end in np.flatnonzero(done == 1):
        z[start:end + 1] = r[start:end + 1][::-1].cumsum()[::-1]
        start = end + 1
    assert start == len(r), "last game is not terminated with done=1"
    return z


def make_targets(examples):
    """examples dict -> (boards u64, pi float32 (N,4), z float32 (N,))."""
    visits = examples["visits"].astype(np.float32)
    pi = visits / visits.sum(axis=1, keepdims=True)
    z = np.log1p(future_scores(examples["reward"], examples["done"])) / Z_SCALE
    return examples["board"].copy(), pi, z.astype(np.float32)


class ReplayBuffer:
    """Flat FIFO of (board, pi, z), capped at `capacity` newest positions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.board = np.empty(0, dtype=np.uint64)
        self.pi = np.empty((0, 4), dtype=np.float32)
        self.z = np.empty(0, dtype=np.float32)

    def add(self, boards, pi, z):
        self.board = np.concatenate([self.board, boards])[-self.capacity:]
        self.pi = np.concatenate([self.pi, pi])[-self.capacity:]
        self.z = np.concatenate([self.z, z])[-self.capacity:]

    def __len__(self):
        return len(self.board)

    def sample(self, batch_size, rng):
        idx = rng.integers(len(self.board), size=batch_size)
        return self.board[idx], self.pi[idx], self.z[idx]

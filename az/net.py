"""Policy+value network and board encoding.

Boards are 2048 bitboards (uint64, 16 nibbles of log2 tile value) encoded
as 16 one-hot planes over the 4x4 grid. The value head predicts
log1p(future merge score) / Z_SCALE, so an immediately lost position has
value 0 and strong late-game positions sit around 1.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Z_SCALE = 12.0  # log1p(score): 2048-game ~ 10.4, 4096-game ~ 11.1


class PolicyValueNet(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.conv1 = nn.Conv2d(16, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.fc = nn.Linear(ch * 16, 256)
        self.policy_head = nn.Linear(256, 4)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.flatten(1)))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


def nibbles_torch(boards_u64: np.ndarray, device) -> torch.Tensor:
    """(N,) uint64 bitboards -> (N, 16) int64 nibble values on device."""
    b = torch.from_numpy(boards_u64.view(np.int64)).to(device)
    shifts = torch.arange(16, device=device, dtype=torch.int64) * 4
    return (b[:, None] >> shifts[None, :]) & 0xF


def encode_boards(boards_u64: np.ndarray, device) -> torch.Tensor:
    """(N,) uint64 bitboards -> (N, 16, 4, 4) float32 one-hot planes."""
    nib = nibbles_torch(boards_u64, device)
    x = F.one_hot(nib, 16).permute(0, 2, 1).float()
    return x.reshape(-1, 16, 4, 4)


def save_checkpoint(net: PolicyValueNet, path):
    torch.save({"state_dict": net.state_dict()}, path)


def load_checkpoint(path, device) -> PolicyValueNet:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = PolicyValueNet().to(device)
    net.load_state_dict(ckpt["state_dict"])
    return net

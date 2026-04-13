"""
visualize_uv_quiver_2d.py
=========================
Load trained OT-CFM + Sinkformer 2d checkpoints; plot theory v vs pred u (quiver).
Core figure logic lives in uv_quiver_2d_plot.py.

Run:
  python experiments/visualize_uv_quiver_2d.py --ckpt-dir experiments/results/sinkformer_2d_b64_v2
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _EXP)

from sinkformer_2d_pairer import SinkPairer
from uv_quiver_2d_plot import save_uv_quiver_figure


class FlowNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t.view(-1, 1).expand(x.shape[0], 1)], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', type=str, default='experiments/results/sinkformer_2d_b64_v2')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--d-sink', type=int, default=16, dest='d_sink')
    ap.add_argument('--sink-iters', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--soft-pair', action='store_true', help='Use soft P in Sinkformer panel (default: Hungarian STE)')
    ap.add_argument(
        '--per-pair-t', action='store_true',
        help='Sample t independently per pair; default is one shared t',
    )
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    ckpt = args.ckpt_dir
    hard_st = not args.soft_pair

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net_ot = FlowNet(args.hidden).to(device)
    net_sf = FlowNet(args.hidden).to(device)
    pairer = SinkPairer(args.d_sink, n_sink_iters=args.sink_iters).to(device)

    net_ot.load_state_dict(torch.load(os.path.join(ckpt, 'net_ot.pt'), map_location=device))
    net_sf.load_state_dict(torch.load(os.path.join(ckpt, 'net_sf.pt'), map_location=device))
    pairer.load_state_dict(torch.load(os.path.join(ckpt, 'pairer.pt'), map_location=device))

    out = os.path.join(ckpt, 'uv_quiver_ot_vs_sf.png')
    save_uv_quiver_figure(
        out,
        net_ot,
        net_sf,
        pairer,
        device,
        args.batch,
        args.seed,
        hard_st,
        per_pair_t=args.per_pair_t,
    )
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

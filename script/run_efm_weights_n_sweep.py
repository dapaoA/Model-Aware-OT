"""
Run analyze_efm_weights_over_t.py for n = 2^5, 2^6, ..., 2^12 (32 to 4096).
Saves main figure and top-k images to output_dir/n/ subfolders.
"""

import subprocess
import sys
from pathlib import Path


def main():
    output_dir = Path("exp/experiment_results")
    for exp in range(5, 16):
        n = 2 ** exp
        cmd = [
            sys.executable,
            "script/analyze_efm_weights_over_t.py",
            "--n", str(n),
            "--output_dir", str(output_dir),
            "--save_topk",
        ]
        print(f"\n{'='*60}\nRunning n={n}\n{'='*60}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

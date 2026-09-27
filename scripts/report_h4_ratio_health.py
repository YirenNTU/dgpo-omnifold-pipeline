#!/usr/bin/env python3
"""Recompute the read-only ratio report from a trusted local export; no training."""
import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evenet_dgpo'))
from RL.DGPO_neutrino.omnifold_ztautau.ratio_health import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if not (args.directory/'COMPLETE').is_file():
        parser.error('Export incomplete; do not use this artifact')
    bundle = torch.load(args.directory/'best_classifier_and_test.pt', map_location='cpu', weights_only=True)
    if bundle['schema'] != 'h4-ratio-health-v1':
        parser.error('Unknown export schema')
    print(json.dumps(report(bundle['truth_logits'],bundle['gen_logits'],
                            bundle['truth_topology'],bundle['gen_topology']), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

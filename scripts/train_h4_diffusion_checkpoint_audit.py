#!/usr/bin/env python3
"""Fresh matched H4 audit of supervised diffusion checkpoints; user launches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml
from train_h4_checkpoint_audit import configuration as audit_configuration
from train_dgpo_token_film import ROOT, BASE
from train_neutrino_backend import read_overlay_yaml

DIRECTORIES = {
    "global": "diffusion_global_film_long",
}


def inspect_source(path, arm):
    import torch
    path = Path(path).resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("state_dict", {})
    if not state:
        raise ValueError("Checkpoint has no raw state_dict")
    step, epoch = payload.get("global_step"), payload.get("epoch")
    if type(step) is not int or step < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("Missing valid checkpoint epoch/global_step")
    readout = {k: v for k, v in state.items() if "TruthGeneration.visible_conditioning.token_readout." in k}
    if readout:
        raise ValueError("Expected global FiLM checkpoint without output residual")
    return {"path": str(path), "epoch": epoch, "global_step": step}


def latest_source(directory, arm):
    paths = sorted({p.resolve() for p in Path(directory).glob("*.ckpt") if p.is_file()})
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {directory}")
    records = [inspect_source(p, arm) for p in paths]
    return max(records, key=lambda x: (x["epoch"], x["global_step"], x["path"]))


def configuration(arm, metadata, output):
    cfg = audit_configuration(Path(metadata["path"]), output, metadata=metadata)
    policy = read_overlay_yaml(ROOT / "config/train_diffusion_global_film_long.yaml")
    # Copy policy architecture only. Preserve the existing classifier, data,
    # normalization, fit budget, pretrained classifier source and test split.
    cfg["network"]["VisibleConditioning"]["diffusion_token_readout"] = (
        policy["network"]["VisibleConditioning"]["diffusion_token_readout"])
    cfg["options"]["Training"]["pretrain_model_load_path"] = None
    label = "global FiLM"
    cfg["logger"]["wandb"].update(
        run_name=f"Does diffusion improve fresh H4 separation? | {label} | epoch {metadata['epoch']}",
        group="H4 supervised diffusion audits",
        tags=["FreshAudit", "MatchedFold", "RawWeights", "SixteenGPU", "NoPolicyUpdate", arm],
    )
    cfg["logger"]["local"]["name"] = f"h4-diffusion-{arm}-audit"
    ex = cfg["experiment"]
    ex.update(arm=f"fresh_h4_supervised_{arm}", source_stage="raw_supervised_diffusion",
              intervention="fresh_classifier_on_supervised_diffusion_samples",
              single_question="Does the supervised diffusion checkpoint reduce held-out fresh-H4 separation?",
              comparison_run=None, supersedes_unmatched_audit=None)
    cfg["nersc"]["reproducibility"].update(
        source_checkpoint=metadata["path"],
        note="Pinned raw supervised policy; fresh H4 from original pretrained backbone. Same fold identities, K1/DDIM20 and independent early-stop/test splits as the matched checkpoint audit. No saved reward reuse or policy updates.")
    cfg["nersc"]["execution"]["command"] = (
        f"shifter python3 -u scripts/train_h4_diffusion_checkpoint_audit.py --arm {arm}")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=DIRECTORIES, default="global")
    parser.add_argument("--checkpoint", type=Path, help="Explicit source; otherwise select latest by saved epoch/step")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = Path("/pscratch/sd/y/yiren/Ztautau") / DIRECTORIES[args.arm] / "checkpoints"
    metadata = ({"path": str(args.checkpoint or source / "last.ckpt"), "epoch": 0, "global_step": 0}
                if args.dry_run else inspect_source(args.checkpoint, args.arm)
                if args.checkpoint else latest_source(source, args.arm))
    output = args.output or source.parent.parent / f"h4_{DIRECTORIES[args.arm]}_epoch{metadata['epoch']}_audit"
    cfg = configuration(args.arm, metadata, output)
    if args.dry_run:
        cfg["experiment"]["checkpoint_metadata_verified"] = False
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    print(json.dumps({"source": metadata, "arm": args.arm, "workers": cfg["platform"]["number_of_workers"],
                      "output": str(output), "classifier": "fresh H4; best validation BCE; disjoint test"}, indent=2), flush=True)
    with tempfile.TemporaryDirectory(prefix="h4-diffusion-audit-") as temp:
        overlay = Path(temp) / "audit.yaml"
        overlay.write_text(yaml.safe_dump(cfg, sort_keys=False))
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts/train_neutrino_backend.py"),
                        "--backend", "dgpo-evenet", "--base-config", str(BASE),
                        "--overlay-config", str(overlay), "--", "--ray-dir",
                        cfg["nersc"]["ray"]["results_dir"]], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

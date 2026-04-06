"""
Train -> Validate -> Promote pipeline for the Paris 15e valuation model.

Usage:
    python pipeline.py retrain              # full pipeline: train + validate + promote
    python pipeline.py train                # train a candidate model
    python pipeline.py validate             # validate the candidate
    python pipeline.py promote              # promote candidate to live (with backup)
    python pipeline.py rollback             # restore previous live model

The candidate model is never served directly. It must pass validation
before it can replace the live model. This prevents accidentally
deploying a broken model to production.
"""

import argparse
import os
import shutil
import subprocess
import sys

LIVE_MODEL = "artifacts/model.json"
CANDIDATE = "artifacts/candidate.json"
BACKUP = "artifacts/model.prev.json"
DEFAULT_CSV = "data/dvf.csv"


def run(cmd):
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFailed (exit {result.returncode}). Pipeline stopped.")
        sys.exit(result.returncode)


def train(csv_path):
    print("\n[1/3] Training candidate model...")
    run([sys.executable, "training/train_model.py", "--csv", csv_path, "--out", CANDIDATE])


def validate():
    print("\n[2/3] Validating candidate model...")
    if not os.path.exists(CANDIDATE):
        print(f"No candidate model found at {CANDIDATE}. Run 'python pipeline.py train' first.")
        sys.exit(1)
    run([sys.executable, "training/validate_model.py", "--model", CANDIDATE])


def promote():
    print("\n[3/3] Promoting candidate to live...")
    if not os.path.exists(CANDIDATE):
        print(f"No candidate model found at {CANDIDATE}. Run 'python pipeline.py train' first.")
        sys.exit(1)
    if os.path.exists(LIVE_MODEL):
        shutil.copy2(LIVE_MODEL, BACKUP)
        print(f"  Previous model backed up to {BACKUP}")
    shutil.move(CANDIDATE, LIVE_MODEL)
    print(f"  Candidate promoted to {LIVE_MODEL}")


def rollback():
    if not os.path.exists(BACKUP):
        print("No previous model to roll back to.")
        sys.exit(1)
    shutil.copy2(BACKUP, LIVE_MODEL)
    print(f"Rolled back to previous model.")


def retrain(csv_path):
    train(csv_path)
    validate()
    promote()
    print("\nPipeline complete — new model is live.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paris 15e model pipeline")
    parser.add_argument("command", choices=["retrain", "train", "validate", "promote", "rollback"])
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to DVF CSV (for train/retrain)")
    args = parser.parse_args()

    if args.command == "retrain":
        retrain(args.csv)
    elif args.command == "train":
        train(args.csv)
    elif args.command == "validate":
        validate()
    elif args.command == "promote":
        promote()
    elif args.command == "rollback":
        rollback()

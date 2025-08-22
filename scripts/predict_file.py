#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import requests

from heart_risk.io_utils import write_submission
from heart_risk.predictor import PredictorService


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--test-file", required=True, type=str, help="Path to test CSV")
	parser.add_argument("--output-file", required=True, type=str, help="Path to output submission.csv")
	parser.add_argument("--artifacts-dir", required=False, type=str, default="./artifacts", help="Artifacts dir (local mode)")
	parser.add_argument("--use-api", action="store_true", help="Use running API instead of local prediction")
	parser.add_argument("--api-url", required=False, type=str, help="API base URL, e.g. http://127.0.0.1:8010")
	return parser.parse_args()


def predict_api(test_path: Path, base_url: str, out_path: Path) -> None:
	with open(test_path, "rb") as f:
		files = {"file": (test_path.name, f, "text/csv")}
		resp = requests.post(f"{base_url}/predict-upload", files=files, timeout=120)
		resp.raise_for_status()
		data = resp.json()
	rows = data["predictions"]
	ids = [r["id"] for r in rows]
	preds = [r["prediction"] for r in rows]
	write_submission(ids, preds, out_path)
	print(f"Saved submission to {out_path}")


def main() -> None:
	args = parse_args()
	test_path = Path(args.test_file)
	out_path = Path(args.output_file)
	if args.use_api:
		if not args.api_url:
			print("Error: --api-url is required when --use-api is set")
			return
		predict_api(test_path, args.api_url.rstrip("/"), out_path)
	else:
		print("Running local prediction...")
		artifacts_dir = Path(args.artifacts_dir)
		model_path = artifacts_dir / "best_pipeline.pkl"

		if not model_path.exists():
			print(f"Error: Model artifact not found at {model_path}")
			print("Please run the 'notebooks/comprehensive_analysis.ipynb' first.")
			return

		predictor = PredictorService(model_path=model_path)
		df = pd.read_csv(test_path)
		probs = predictor.predict_df(df)
		ids = df["id"].tolist() if "id" in df.columns else list(range(len(probs)))
		write_submission(ids, probs, out_path)
		print(f"Saved submission to {out_path}")


if __name__ == "__main__":
	main()

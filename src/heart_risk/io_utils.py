from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def read_csv_strict(path: str | Path) -> pd.DataFrame:
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"CSV file not found: {path}")
	return pd.read_csv(path)


def write_submission(ids: Iterable, preds: Iterable, out_path: str | Path) -> None:
	out_path = Path(out_path)
	df = pd.DataFrame({"id": list(ids), "prediction": list(preds)})
	df.to_csv(out_path, index=False)

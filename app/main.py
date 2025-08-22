from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles

import io
import csv
import json
from uuid import uuid4

from src.heart_risk.predictor import PredictorService
from src.heart_risk.io_utils import read_csv_strict

# --- Конфигурация ---
# Создаем директорию для временных файлов, если ее нет
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# Загружаем модель при старте приложения
PIPELINE_PATH = Path("artifacts/best_pipeline.pkl")
if not PIPELINE_PATH.exists():
    raise FileNotFoundError(
        "Model artifact 'best_pipeline.pkl' not found. "
        "Please run 'notebooks/comprehensive_analysis.ipynb' to generate it."
    )
predictor = PredictorService(model_path=PIPELINE_PATH)

app = FastAPI()

# Ensure tmp dir exists before mounting
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/tmp", StaticFiles(directory=str(TMP_DIR)), name="tmp")

# Mount static
STATIC_DIR = Path("app/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory="app/templates")

_predictor: PredictorService | None = None


def get_predictor() -> PredictorService:
	global _predictor
	if _predictor is None:
		try:
			_predictor = PredictorService(PIPELINE_PATH)
		except Exception as exc:
			raise HTTPException(status_code=500, detail=f"Failed to load artifacts: {exc}")
	return _predictor


@app.get("/health")
def health() -> dict:
	return {"status": "OK"}


@app.get("/")
async def index(request: Request):
	return templates.TemplateResponse("start_form.html", {"request": request})


@app.post("/predict-path")
async def predict_by_path(path: str) -> JSONResponse:
	predictor = get_predictor()
	try:
		df = read_csv_strict(path)
		preds = predictor.predict_df(df)
		ids = df["id"].tolist() if "id" in df.columns else list(range(len(preds)))
		rows = [{"id": int(i), "prediction": float(p)} for i, p in zip(ids, preds)]
		return JSONResponse({"predictions": rows, "count": len(rows)})
	except Exception as exc:
		raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict-upload")
async def predict_upload(file: UploadFile | None = File(None)) -> JSONResponse:
	if file is None:
		raise HTTPException(status_code=400, detail="Файл не передан. Загрузите CSV-файл в поле 'file'.")
	if not file.filename.lower().endswith(".csv"):
		raise HTTPException(status_code=400, detail="Неверный формат. Ожидается CSV файл.")

	save_path = TMP_DIR / file.filename
	with open(save_path, "wb") as f:
		f.write(await file.read())
	predictor = get_predictor()
	try:
		df = read_csv_strict(save_path)
		preds = predictor.predict_df(df)
		ids = df["id"].tolist() if "id" in df.columns else list(range(len(preds)))
		rows = [{"id": int(i), "prediction": float(p)} for i, p in zip(ids, preds)]

		# Генерируем token, сохраняем json и переименовываем исходный CSV к token.csv
		token = uuid4().hex
		csv_path = TMP_DIR / f"{token}.csv"
		try:
			save_path.rename(csv_path)
		except Exception:
			csv_path = save_path  # если переименование не удалось, оставим как есть

		json_path = TMP_DIR / f"{token}.json"
		with open(json_path, "w", encoding="utf-8") as jf:
			json.dump({"predictions": rows, "count": len(rows), "csv_path": str(csv_path)}, jf, ensure_ascii=False)

		return JSONResponse({
			"token": token,
			"predictions": rows[:5],
			"count": len(rows),
			"results_url": f"/results/{token}",
			"download_url": f"/results/csv/{token}"
		})
	finally:
		pass


@app.get("/results/{token}", response_class=HTMLResponse)
async def results_page(request: Request, token: str):
	json_path = TMP_DIR / f"{token}.json"
	if not json_path.exists():
		raise HTTPException(status_code=404, detail="Результаты не найдены или ссылка устарела")
	with open(json_path, "r", encoding="utf-8") as jf:
		data = json.load(jf)
	predictions = data.get("predictions", [])
	count = data.get("count", len(predictions))
	return templates.TemplateResponse("res_form.html", {
		"request": request,
		"predictions": predictions,
		"count": count,
		"token": token
	})


@app.get("/results/{token}.csv")
async def results_csv(token: str) -> StreamingResponse:
	json_path = TMP_DIR / f"{token}.json"
	if not json_path.exists():
		raise HTTPException(status_code=404, detail="Результаты не найдены или ссылка устарела")
	with open(json_path, "r", encoding="utf-8") as jf:
		data = json.load(jf)
	predictions = data.get("predictions", [])

	buf = io.StringIO()
	w = csv.writer(buf)
	w.writerow(["id", "prediction"])
	for row in predictions:
		w.writerow([row.get("id"), row.get("prediction")])
	buf.seek(0)

	headers = {"Content-Disposition": f"attachment; filename=predictions_{token}.csv"}
	return StreamingResponse(buf, media_type="text/csv", headers=headers)


@app.get("/results/csv/{token}")
async def results_csv_alt(token: str) -> StreamingResponse:
	json_path = TMP_DIR / f"{token}.json"
	if not json_path.exists():
		raise HTTPException(status_code=404, detail="Результаты не найдены или ссылка устарела")
	with open(json_path, "r", encoding="utf-8") as jf:
		data = json.load(jf)
	predictions = data.get("predictions", [])

	buf = io.StringIO()
	w = csv.writer(buf)
	w.writerow(["id", "prediction"])
	for row in predictions:
		w.writerow([row.get("id"), row.get("prediction")])
	buf.seek(0)

	headers = {"Content-Disposition": f"attachment; filename=predictions_{token}.csv"}
	return StreamingResponse(buf, media_type="text/csv", headers=headers)


@app.get("/results/json/{token}")
async def results_json(token: str) -> JSONResponse:
	json_path = TMP_DIR / f"{token}.json"
	if not json_path.exists():
		raise HTTPException(status_code=404, detail="Результаты не найдены или ссылка устарела")
	with open(json_path, "r", encoding="utf-8") as jf:
		data = json.load(jf)
	return JSONResponse(data)


@app.delete("/results/{token}")
async def results_delete(token: str) -> JSONResponse:
	json_path = TMP_DIR / f"{token}.json"
	csv_path = TMP_DIR / f"{token}.csv"
	deleted = {"json": False, "csv": False}
	if json_path.exists():
		try:
			json_path.unlink()
			deleted["json"] = True
		except Exception:
			pass
	if csv_path.exists():
		try:
			csv_path.unlink()
			deleted["csv"] = True
		except Exception:
			pass
	return JSONResponse({"ok": True, "deleted": deleted})

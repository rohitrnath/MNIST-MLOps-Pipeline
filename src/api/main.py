import os
import yaml
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


class PredictRequest(BaseModel):
	# image as flattened list (784) or nested 28x28 list
	image: list


def load_params() -> dict:
	with open("params.yaml", "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


@app.on_event("startup")
async def startup_event():
	params = load_params()
	mlf = params["mlflow"]
	api_cfg = params.get("api", {})
	mn = params.get("mnist", {})
	app.state.norm_mean = float(mn.get("normalize_mean", 0.1307))
	app.state.norm_std = float(mn.get("normalize_std", 0.3081))
	os.environ.setdefault("MLFLOW_TRACKING_URI", mlf["tracking_uri"])
	model_name = mlf["registered_model_name"]
	stage = api_cfg.get("stage_to_load", "Production")
	uri = f"models:/{model_name}/{stage}"
	app.state.model = mlflow.pyfunc.load_model(uri)
	# mount static if available
	if os.path.isdir("static"):
		app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root_index():
	index_path = os.path.join("static/", "index.html")
	if os.path.exists(index_path):
		return FileResponse(index_path)
	return {"message": "MNIST API. POST /predict with image list."}


@app.post("/predict")
async def predict(req: PredictRequest):
	arr = np.array(req.image, dtype=np.float32)
	if arr.ndim == 1:
		if arr.size != 28 * 28:
			return {"error": f"Invalid flattened length {arr.size}, expected 784"}
		arr = arr.reshape(28, 28)
	elif arr.ndim == 2:
		if arr.shape != (28, 28):
			return {"error": f"Invalid image shape {arr.shape}, expected (28, 28)"}
	else:
		return {"error": "Invalid image shape"}
	# scale to [0,1] if needed
	if float(np.max(arr)) > 1.5:
		arr = arr / 255.0
	# normalize using training stats
	arr = (arr - app.state.norm_mean) / app.state.norm_std
	# build batch for model: (N,C,H,W)
	batch = np.expand_dims(arr, axis=(0, 1)).astype(np.float32)
	pred = app.state.model.predict(batch)
	pred_np = np.array(pred)
	if pred_np.ndim == 2 and pred_np.shape[0] == 1:
		pred_class = int(np.argmax(pred_np, axis=1)[0])
	elif pred_np.ndim == 1:
		pred_class = int(np.argmax(pred_np))
	else:
		# fallback for lists
		try:
			pred_class = int(np.argmax(np.array(pred)))
		except Exception:
			return {"error": "Unexpected prediction output format"}
	return {"prediction": pred_class}

import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
import pandas as pd
import mlflow
import mlflow.pytorch as mlflow_pytorch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


def load_params(params_path: str) -> dict:
	with open(params_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def _normalize_metric_name(name: str) -> str:
	return "".join(c.lower() if c.isalnum() else "_" for c in str(name)).strip("_")


def _log_mlflow_metrics(prefix: str, stats: dict, step: int) -> None:
	if not isinstance(stats, dict):
		return
	for k, v in stats.items():
		try:
			val = float(v)
		except (TypeError, ValueError):
			continue
		mlflow.log_metric(f"{prefix}_{_normalize_metric_name(k)}", val, step=step)


def _get_accuracy(stats: dict) -> float:
	if not isinstance(stats, dict):
		return 0.0
	for key in [
		"Accuracy",
		"accuracy",
		"acc",
		"top1accuracy",
		"top1_accuracy",
	]:
		if key in stats:
			try:
				return float(stats[key])
			except (TypeError, ValueError):
				pass
	return 0.0


class ParquetMNIST(Dataset):
	def __init__(self, path: str):
		# Load parquet with pandas/pyarrow to avoid SparkContext usage in workers
		df = pd.read_parquet(path, engine="pyarrow")
		images = []
		labels = []
		for _, row in df.iterrows():
			img = np.array(row["image"], dtype=np.float32).reshape(1, int(row["height"]), int(row["width"]))
			images.append(img)
			labels.append(int(row["label"]))
		self.images = images
		self.labels = labels
		del df

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def train_loader_creator(config, batch_size):
	return DataLoader(config["train_dataset"], batch_size=batch_size, shuffle=True)


def test_loader_creator(config, batch_size):
	return DataLoader(config["test_dataset"], batch_size=batch_size, shuffle=False)


def model_creator(config):
	return Net()


def optimizer_creator(model, config):
	opt_name = str(config.get("optimizer", "adam")).lower()
	lr = float(config["lr"])
	if opt_name == "sgd":
		momentum = float(config.get("momentum", 0.9))
		return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	if opt_name == "adamw":
		weight_decay = float(config.get("weight_decay", 0.01))
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	# default adam
	return torch.optim.Adam(model.parameters(), lr=lr)


def scheduler_creator(optimizer, config):
	sch_name = str(config.get("scheduler", "none")).lower()
	if sch_name == "step":
		step_size = int(config.get("step_size", 10))
		gamma = float(config.get("gamma", 0.1))
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
	if sch_name == "cosine":
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config.get("t_max", 10)))
	return None


def _extract_state_dict(ckpt: dict):
	# Try common layouts produced by Orca Estimator
	if isinstance(ckpt, dict):
		models = ckpt.get("models")
		if models is not None:
			# list of state_dicts or modules
			if isinstance(models, list) and len(models) > 0:
				m0 = models[0]
				if isinstance(m0, dict):
					return m0
				if hasattr(m0, "state_dict"):
					return m0.state_dict()
			# dict of rank->state_dict/module
			if isinstance(models, dict) and len(models) > 0:
				first = next(iter(models.values()))
				if isinstance(first, dict):
					return first
				if hasattr(first, "state_dict"):
					return first.state_dict()
		# other common keys
		for key in ["state_dict", "model_state_dict", "model"]:
			val = ckpt.get(key)
			if isinstance(val, dict):
				return val
	raise RuntimeError("Unable to extract state_dict from checkpoint")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--params", type=str, default="params.yaml")
	args = parser.parse_args()

	params = load_params(args.params)
	mn = params["mnist"]
	mlf = params["mlflow"]
	# Training configuration from params.yaml only
	train_section = params.get("train", {})
	train_cfg = {
		"lr": float(train_section.get("lr", 1e-3)),
		"optimizer": str(train_section.get("optimizer", "adam")).lower(),
		"momentum": float(train_section.get("momentum", 0.9)),
		"weight_decay": float(train_section.get("weight_decay", 0.0)),
		"scheduler": str(train_section.get("scheduler", "none")).lower(),
		"step_size": int(train_section.get("step_size", 10)),
		"gamma": float(train_section.get("gamma", 0.1)),
		"model_name": str(train_section.get("model_name", "Net"))
	}

	# Prefer Orca-bundled Spark; unset external SPARK_HOME if set
	os.environ.pop("SPARK_HOME", None)

	# Detect GPUs on Linux servers
	gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
	backend = "torch_distributed"
	workers_per_node = max(1, gpu_count) if gpu_count > 0 else 1

	init_orca_context(cluster_mode="local", cores=8, memory="8g", num_nodes=1)

	train_path = os.path.join(mn["output_dir"], "train.parquet")
	test_path = os.path.join(mn["output_dir"], "test.parquet")
	train_ds = ParquetMNIST(train_path)
	test_ds = ParquetMNIST(test_path)

	mlflow.set_tracking_uri(mlf["tracking_uri"])
	mlflow.set_experiment(mlf["experiment_name"])

	best_metric = None
	best_run_id = None
	best_cfg = None

	# Ensure artifacts directory exists
	os.makedirs("artifacts", exist_ok=True)

	with mlflow.start_run(description=mlf.get("run_description", "")) as run:
		# Build estimator kwargs to optionally include scheduler
		est_kwargs = dict(
			model=model_creator,
			optimizer=optimizer_creator,
			loss=nn.NLLLoss(),
			metrics=[Accuracy()],
			use_tqdm=True,
			workers_per_node=workers_per_node,
			config={
				"lr": train_cfg["lr"],
				"optimizer": train_cfg["optimizer"],
				"momentum": train_cfg["momentum"],
				"weight_decay": train_cfg["weight_decay"],
				"scheduler": train_cfg["scheduler"],
				"step_size": train_cfg["step_size"],
				"gamma": train_cfg["gamma"],
				"model name": train_cfg["model_name"],
				"train_dataset": train_ds,
				"test_dataset": test_ds,
			},
		)
		if str(train_cfg["scheduler"]).lower() != "none":
			est_kwargs["scheduler"] = scheduler_creator
		est = Estimator.from_torch(**est_kwargs)

		# Log run configuration
		mlflow.log_params({
			"lr": train_cfg["lr"],
			"optimizer": train_cfg["optimizer"],
			"momentum": train_cfg["momentum"],
			"weight_decay": train_cfg["weight_decay"],
			"scheduler": train_cfg["scheduler"],
			"step_size": train_cfg["step_size"],
			"gamma": train_cfg["gamma"],
			"batch_size": mn["batch_size"],
			"epochs": mn["epochs"],
			"workers_per_node": workers_per_node,
		})

		# Per-epoch training and evaluation to log metrics
		best_metric = None
		for epoch in range(int(mn["epochs"])):
			train_stats = est.fit(data=train_loader_creator, epochs=1, batch_size=mn["batch_size"])
			# Log all numeric training metrics for this epoch
			train_last = train_stats[-1] if isinstance(train_stats, list) else train_stats
			_log_mlflow_metrics("train", train_last, step=epoch + 1)

			eval_stats = est.evaluate(data=test_loader_creator, batch_size=mn["batch_size"])
			# Log all numeric evaluation metrics
			_log_mlflow_metrics("test", eval_stats, step=epoch + 1)

			acc = _get_accuracy(eval_stats)
			if best_metric is None or acc > best_metric:
				best_metric = acc
				best_run_id = run.info.run_id
				best_cfg = {"lr": train_cfg["lr"], "model_name": train_cfg["model_name"]}

		# Save estimator checkpoint and extract model weights
		ckpt_path = os.path.join("artifacts", f"mnist_ckpt_{int(time.time())}.pt")
		est.save(ckpt_path)
		ckpt = torch.load(ckpt_path, map_location="cpu")
		state = _extract_state_dict(ckpt)
		model = Net()
		model.load_state_dict(state)
		model.eval()
		# Log as MLflow PyTorch Model under 'model'
		mlflow_pytorch.log_model(model, artifact_path="model")
		# Record a tag and tiny marker to help the registrar find it
		mlflow.set_tag("logged_model_artifact", "model")
		open("model_marker.txt", "w", encoding="utf-8").write("model artifact logged")
		with open("model_marker.txt", "w") as f:
			f.write(str(summary(model, input_size=(1, 1, 28, 28))))
		mlflow.log_artifact("model_marker.txt")

		est.shutdown()

	if best_run_id is not None:
		mlflow.set_tag("best_run_id", best_run_id)
		print({"best_run_id": best_run_id, "best_metric": best_metric, "best_cfg": best_cfg})

	stop_orca_context()


if __name__ == "__main__":
	main()

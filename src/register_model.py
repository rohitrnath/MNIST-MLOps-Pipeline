import argparse
import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient


def load_params(params_path: str) -> dict:
	with open(params_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def _model_uri_if_exists(client: MlflowClient, run_id: str):
	try:
		children = client.list_artifacts(run_id, "model")
		if any(c.path.endswith("MLmodel") for c in children):
			return f"runs:/{run_id}/model"
		return None
	except Exception:
		return None


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--params", type=str, default="params.yaml")
	args = parser.parse_args()

	params = load_params(args.params)
	mlf = params["mlflow"]

	mlflow.set_tracking_uri(mlf["tracking_uri"])
	experiment = mlflow.get_experiment_by_name(mlf["experiment_name"])
	if experiment is None:
		raise RuntimeError("Experiment not found. Run training first.")

	client = MlflowClient()
	# Choose best by accuracy descending
	runs = client.search_runs([experiment.experiment_id], order_by=["metrics.accuracy DESC", "attributes.start_time DESC"], max_results=200)
	if not runs:
		raise RuntimeError("No runs found in experiment.")

	model_name = mlf["registered_model_name"]
	chosen = None
	model_uri = None
	for r in runs:
		uri = _model_uri_if_exists(client, r.info.run_id)
		if uri:
			chosen = r
			model_uri = uri
			break

	if chosen is None or model_uri is None:
		raise RuntimeError("No suitable run contains a logged MLflow model under 'model/'. Re-run training.")

	version = mlflow.register_model(model_uri=model_uri, name=model_name)
	print(f"Registered {model_name} v{version.version} from run {chosen.info.run_id}")

	# Use stages only
	stage = mlf.get("register_stage", "Staging")
	client.transition_model_version_stage(name=model_name, version=version.version, stage=stage, archive_existing_versions=False)
	print(f"Transitioned to {stage}")
	if bool(mlf.get("promote_to_production", False)) and stage != "Production":
		client.transition_model_version_stage(name=model_name, version=version.version, stage="Production", archive_existing_versions=False)
		print("Promoted to Production")


if __name__ == "__main__":
	main()

import argparse
import json
import os
import yaml
from pathlib import Path
from pyspark.sql import SparkSession


def load_params(params_path: str) -> dict:
	with open(params_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def psi(expected: dict, actual: dict) -> float:
	# Population Stability Index for categorical distribution
	bins = sorted(set(expected.keys()) | set(actual.keys()))
	import math
	value = 0.0
	for b in bins:
		e = max(1e-6, float(expected.get(b, 0.0)))
		a = max(1e-6, float(actual.get(b, 0.0)))
		value += (a - e) * (0.0 if a == 0.0 or e == 0.0 else math.log(a / e))
	return float(value)


def distribution_from_parquet(spark: SparkSession, path: str) -> dict:
	df = spark.read.parquet(path)
	rows = df.groupBy("label").count().collect()
	total = sum(r["count"] for r in rows)
	return {int(r["label"]): r["count"] / total for r in rows}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--params", type=str, default="params.yaml")
	parser.add_argument("--new-data", type=str, required=True)
	args = parser.parse_args()

	params = load_params(args.params)
	base_path = os.path.join(params["mnist"]["output_dir"], "baseline_distribution.json")
	with open(base_path, "r", encoding="utf-8") as f:
		baseline = {int(k): float(v) for k, v in json.load(f).items()}

	spark = SparkSession.builder.appName("mnist-drift").getOrCreate()
	current = distribution_from_parquet(spark, args.new_data)
	drift = psi(baseline, current)
	print(json.dumps({"psi": drift, "baseline": baseline, "current": current}, indent=2))

	threshold = float(params["retrain"]["drift_threshold_psi"])
	if drift >= threshold:
		print("ALERT: Drift detected")
		open("drift_alert.txt", "w", encoding="utf-8").write(f"PSI={drift}")

	spark.stop()


if __name__ == "__main__":
	main()

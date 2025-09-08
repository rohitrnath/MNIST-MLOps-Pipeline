import argparse
import json
import os
import random
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from torchvision import datasets, transforms


def load_params(params_path: str) -> dict:
	with open(params_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def prepare_output(output_dir: str) -> None:
	Path(output_dir).mkdir(parents=True, exist_ok=True)


def mnist_iter(split: str, normalize_mean: float, normalize_std: float):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((normalize_mean,), (normalize_std,))
	])
	ds = datasets.MNIST(root="./data/raw_mnist", train=(split == "train"), download=True, transform=transform)
	for img_tensor, label in ds:
		arr = img_tensor.numpy().astype(np.float32)
		yield int(label), arr


def write_parquet_in_chunks(ss: SparkSession, records_iter, out_path: str, chunk_size: int, partitions: int):
	first_chunk = True
	buffer_labels = []
	buffer_images = []
	buffer_h = []
	buffer_w = []
	for idx, (label, arr) in enumerate(records_iter, start=1):
		buffer_labels.append(label)
		buffer_images.append(arr.flatten().tolist())
		buffer_h.append(int(arr.shape[1]))
		buffer_w.append(int(arr.shape[2]))
		if len(buffer_labels) >= chunk_size:
			pdf = pd.DataFrame({
				"label": buffer_labels,
				"image": buffer_images,
				"height": buffer_h,
				"width": buffer_w,
			})
			df = ss.createDataFrame(pdf).repartition(partitions)
			mode = "overwrite" if first_chunk else "append"
			df.write.mode(mode).parquet(out_path)
			first_chunk = False
			buffer_labels.clear(); buffer_images.clear(); buffer_h.clear(); buffer_w.clear()
	# Flush remainder
	if buffer_labels:
		pdf = pd.DataFrame({
			"label": buffer_labels,
			"image": buffer_images,
			"height": buffer_h,
			"width": buffer_w,
		})
		df = ss.createDataFrame(pdf).repartition(partitions)
		mode = "overwrite" if first_chunk else "append"
		df.write.mode(mode).parquet(out_path)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--params", type=str, default="params.yaml")
	args = parser.parse_args()

	params = load_params(args.params)
	mn = params["mnist"]

	output_dir = mn["output_dir"]
	prepare_output(output_dir)

	driver_mem = mn.get("spark_driver_memory", "2g")
	ss = (
		SparkSession.builder.appName("mnist-preprocess")
		.config("spark.sql.execution.arrow.pyspark.enabled", "true")
		.config("spark.driver.memory", driver_mem)
		.getOrCreate()
	)

	partitions = int(mn.get("spark_partitions", 8))
	chunk_size = int(mn.get("chunk_size", 2048))

	for split in ["train", "test"]:
		rec_iter = mnist_iter(split, mn["normalize_mean"], mn["normalize_std"])
		out_path = os.path.join(output_dir, f"{split}.parquet")
		write_parquet_in_chunks(ss, rec_iter, out_path, chunk_size, partitions)

	# Baseline distribution from train
	train_df = ss.read.parquet(os.path.join(output_dir, "train.parquet"))
	counts = train_df.groupBy("label").count().orderBy("label").collect()
	total = sum(r["count"] for r in counts)
	baseline = {int(r["label"]): r["count"] / total for r in counts}
	with open(os.path.join(output_dir, "baseline_distribution.json"), "w", encoding="utf-8") as f:
		json.dump(baseline, f, indent=2)

	ss.stop()


if __name__ == "__main__":
	main()

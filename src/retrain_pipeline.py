import argparse
import subprocess
import sys
import yaml


def load_params(params_path: str) -> dict:
	with open(params_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def run(cmd: list):
	print("Running:", " ".join(cmd))
	res = subprocess.run(cmd, check=True)
	return res.returncode


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--params", type=str, default="params.yaml")
	args = parser.parse_args()

	params = load_params(args.params)

	run([sys.executable, "src/preprocess_mnist_spark.py", "--params", args.params])
	run([sys.executable, "src/train_mnist_orca.py", "--params", args.params])
	run([sys.executable, "src/register_model.py", "--params", args.params])


if __name__ == "__main__":
	main()

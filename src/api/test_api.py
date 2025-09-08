import argparse
import json
import numpy as np
import requests
from torchvision import datasets, transforms


def sample_image():
	transform = transforms.ToTensor()
	ds = datasets.MNIST(root="./data/raw_mnist", train=False, download=True, transform=transform)
	img, _ = ds[0]
	arr = img.numpy().reshape(28, 28).astype(float).tolist()
	return arr


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict")
	args = parser.parse_args()

	payload = {"image": sample_image()}
	r = requests.post(args.url, json=payload, timeout=30)
	print(r.status_code, r.text)


if __name__ == "__main__":
	main()

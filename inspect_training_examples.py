import argparse
import random
import json
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Inspect training examples from the dataset")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of examples to display")
    parser.add_argument("--random", action="store_true", help="Select random examples from the dataset")
    args = parser.parse_args()

    dataset = load_dataset("Open-Orca/SlimOrca", split="train", cache_dir="./data")

    if args.random:
        examples = random.sample(list(dataset), min(len(dataset), args.num_examples))
    else:
        examples = dataset.select(range(min(len(dataset), args.num_examples)))

    for i, example in enumerate(examples):
        print(f"Example {i+1}:")
        print(json.dumps(example, indent=2))
        print()

if __name__ == "__main__":
    main() 
import argparse
from custom_dataset_generators.generators import *

parser = argparse.ArgumentParser(description="")
parser.add_argument("--size", type=int, default="1000", help="Size of the dataset")
parser.add_argument("--generator", type=str, default="erdos-renyi",
                    help="Name of generator to be used")
args = parser.parse_args()

generator_name = args.generator
if generator_name == "erdos-renyi":
    generator = ErdosRenyiGenerator

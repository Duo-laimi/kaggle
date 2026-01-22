from trl import SFTConfig
import argparse
from transformers import HfArgumentParser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training_args.yaml")
    args, unknown = parser.parse_known_args()
    return args


def main(args):

    pass



if __name__ == "__main__":
    hf_parser = HfArgumentParser((SFTConfig, ))
    argv = parse_args()
    if argv.config is None:
        argv = hf_parser.parse_args_into_dataclasses()
    else:
        argv = hf_parser.parse_yaml_file(argv.config)[0]
    # argv = parser.parse_args_into_dataclasses()
    main(argv)
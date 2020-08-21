import torch
import argparse
from data.traffic_data_generator import TrafficDataset
import yaml


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def parse_args():
    parser = argparse.ArgumentParser(description="Neural network train script")
    parser.add_argument('--config', required=False, type=str,
                        default='../configs/default_train_config.yml',
                        help='Path to configuration yml file.'
                        )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: {}".format(device))

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


if __name__ == '__main__':
    main()

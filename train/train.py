import torch
import argparse
from data.traffic_data_generator import TrafficDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Neural network train script")
    parser.add_argument('--config', required=False, type=str,
                        default='../config/default_train_config.yml',
                        help='Path to configuration yml file.'
                        )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: {}".format(device))


if __name__ == '__main__':
    main()

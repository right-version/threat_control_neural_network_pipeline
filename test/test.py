import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from data.traffic_data_generator import TrafficDataset
from architectures.autoencoder import Autoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="Neural network inference script")
    parser.add_argument('--config', required=False, type=str,
                        default='../configs/default_train_config.yml',
                        help='Path to configuration yml file.'
                        )
    parser.add_argument('--model-weights', type=str,
                        default="../experiments/experiment_22-08-2020_01-38-26/model_latest.pth",
                        help='Path to model weights')
    parser.add_argument('--output', required=True, type=str,
                        help='Path to result .json file')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model from checkpoint
    print("Loading model from {}".format(args.model_weights))
    checkpoint = torch.load(args.model_weights)
    model = Autoencoder()
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load test data
    dataset_test = TrafficDataset(config["dataset"]["test_dataset"])
    test_loader = DataLoader(dataset_test, batch_size=config["train"]["batch_size"])

    # Evaluate test dataset
    criterion = torch.nn.MSELoss()
    loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            model.eval()
            batch = batch.to(device)

            outputs = model(batch)

            test_loss = criterion(outputs, batch)
            print(test_loss.item())

            loss += test_loss.item()

        loss = loss / len(test_loader)
        print("test loss={}".format(loss))


if __name__ == '__main__':
    main()

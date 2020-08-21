import os
import torch
from torch.utils.data import DataLoader
import argparse
from shutil import copyfile
import yaml
from datetime import datetime

from data.traffic_data_generator import TrafficDataset
from architectures.autoencoder import Autoencoder


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

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create folder for current experiment results
    now = datetime.now()
    prefix = now.strftime("experiment_%d-%m-%Y_%H-%M-%S")
    experiment_path = os.path.join(config["save"]["model"], prefix)
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)

    copyfile(
        args.config,
        os.path.join(
            experiment_path,
            os.path.basename(args.config)
        )
    )

    # Load train/validation data
    dataset_train = TrafficDataset(config["dataset"]["train_dataset"])
    train_loader = DataLoader(dataset_train, batch_size=config["train"]["batch_size"])

    dataset_validation = TrafficDataset(config["dataset"]["validation_dataset"])
    validation_loader = DataLoader(dataset_validation, batch_size=config["train"]["batch_size"])

    # Create model
    model = Autoencoder().to(device)

    # Setup train loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config["train"]["scheduler"]["patience"],
        verbose=True
    )
    criterion = torch.nn.MSELoss()

    # Train loop
    epochs = config["train"]["epochs"]
    for epoch in range(epochs):
        model.train()

        loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch = batch.to(device)
            # Reconstruct input traffic features
            outputs = model(batch)

            #  Compute training reconstruction loss
            train_loss = criterion(outputs, batch)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()

        # Epoch training loss
        loss = loss / len(train_loader)
        lr = get_lr(optimizer)
        scheduler.step(loss)

        # Evaluate model on validation data
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for j, test_batch in enumerate(validation_loader):
                test_batch = test_batch.to(device)

                outputs = model(test_batch)

                tmp_loss = criterion(outputs, test_batch)

                val_loss += tmp_loss.item()
        val_loss = val_loss / len(validation_loader)

        print("epoch : {}/{}, loss = {:.6f}, val_loss = {:.6f}, lr={}".format(epoch + 1, epochs, loss, val_loss, lr))


if __name__ == '__main__':
    main()

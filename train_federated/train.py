from datetime import datetime
from shutil import copyfile
import argparse
import os
import torch
from torch.utils.data import DataLoader
import yaml
import syft as sy
from syft.frameworks.torch.fl import utils
hook = sy.TorchHook(torch)

from data.traffic_data_generator import TrafficDataset
from architectures.autoencoder import Autoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="Federated neural network train script")
    parser.add_argument("--config", required=False, type=str,
                        default="../configs/default_train_config.yml",
                        help="Path to configuration yml file."
                        )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define virtual workers who will emulate real hosts in the network
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")
    secure_worker = sy.VirtualWorker(hook, id="secure_worker")

    with open("../configs/federated_train_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Create folder for current experiment results
    now = datetime.now()
    prefix = now.strftime("federated_experiment_%d-%m-%Y_%H-%M-%S")
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

    # Load Bob's train/validation data
    print("Loading Bob's training/validation data ...")
    bob_dataset_train = TrafficDataset(config["dataset"]["train_dataset_bob"])
    bob_train_loader = DataLoader(bob_dataset_train, batch_size=config["train"]["batch_size"])

    # Load Alice's train/validation data
    print("Loading Alice's training/validation data ...")
    alice_dataset_train = TrafficDataset(config["dataset"]["train_dataset_alice"])
    alice_train_loader = DataLoader(alice_dataset_train, batch_size=config["train"]["batch_size"])

    # Setup Train Loop
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = torch.nn.MSELoss()

    #
    # TODO: This train loop requires massive refactoring ...
    #

    epochs = config["train"]["epochs"]
    print("Start training ...")
    for epoch in range(epochs):
        # Copy base averaged model to workers
        bobs_model = model.copy().send(bob)
        alices_model = model.copy().send(alice)

        # Define workers optimizers
        bobs_opt = torch.optim.Adam(bobs_model.parameters(), lr=config["train"]["lr"])
        alices_opt = torch.optim.Adam(alices_model.parameters(), lr=config["train"]["lr"])

        # Train Bob's Model
        loss_bob = 0
        for i, bobs_batch in enumerate(bob_train_loader):
            bobs_data = bobs_batch.send(bob)
            bobs_opt.zero_grad()
            bobs_pred = bobs_model(bobs_data)
            bobs_loss = criterion(bobs_pred, bobs_data)
            bobs_loss.backward()

            bobs_opt.step()
            loss_bob += bobs_loss.get().item()

        # Train Alice's Model
        loss_alice = 0
        for i, alice_batch in enumerate(alice_train_loader):
            alices_data = alice_batch.send(alice)
            alices_opt.zero_grad()
            alices_pred = alices_model(alices_data)
            alices_loss = criterion(alices_pred, alices_data)
            alices_loss.backward()

            alices_opt.step()
            loss_alice += alices_loss.get().item()

        print("Epoch [{}/{}] Bob: {:.4f} Alice {:.4f}".format(
            epoch + 1, epochs, loss_bob / len(bob_train_loader), loss_alice / len(alice_train_loader)
        ))

        models = {}
        alices_model.move(secure_worker)
        bobs_model.move(secure_worker)

        # Average workers's models weights
        models["bob"] = bobs_model.copy().get()
        models["alice"] = alices_model.copy().get()
        model = utils.federated_avg(models)

        # Save averaged model checkpoint
        if (epoch % config["save"]["every"] == 0) or (epoch == epochs - 1):
            torch.save({
                "model_state_dict": model.cpu().state_dict(),
                "epoch": epoch,
            }, os.path.join(
                experiment_path, "model_epoch{}.pth".format(epoch)
            ))

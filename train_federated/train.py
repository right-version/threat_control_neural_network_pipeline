import torch
import syft as sy
import copy
hook = sy.TorchHook(torch)
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from syft.frameworks.torch.fl import utils

from data.traffic_data_generator import TrafficDataset
from architectures.autoencoder import Autoencoder

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")
    secure_worker = sy.VirtualWorker(hook, id="secure_worker")

    with open("../configs/federated_train_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Load Bob's train/validation data
    print("Loading Bob's training/validation data ...")
    bob_dataset_train = TrafficDataset(config["dataset"]["train_dataset_bob"])
    bob_train_loader = DataLoader(bob_dataset_train, batch_size=config["train"]["batch_size"])

    # Load Alice's train/validation data
    print("Loading Alice's training/validation data ...")
    alice_dataset_train = TrafficDataset(config["dataset"]["train_dataset_alice"])
    alice_train_loader = DataLoader(alice_dataset_train, batch_size=config["train"]["batch_size"])

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = torch.nn.MSELoss()

    # TODO: This train loop requires massive refactoring ...
    epochs = config["train"]["epochs"]
    epochs = 10
    print("Start training ...")
    for epoch in range(epochs):
        bobs_model = model.copy().send(bob)
        alices_model = model.copy().send(alice)

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
            epoch+1, epochs, loss_bob/len(bob_train_loader), loss_alice/len(alice_train_loader)
        ))

        models = {}
        alices_model.move(secure_worker)
        bobs_model.move(secure_worker)

        models["bob"] = bobs_model.copy().get()
        models["alice"] = alices_model.copy().get()
        model = utils.federated_avg(models)

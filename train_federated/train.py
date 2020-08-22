import torch
import syft as sy
import copy
hook = sy.TorchHook(torch)
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from data.traffic_data_generator import TrafficDataset
from architectures.autoencoder import Autoencoder

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    secure_worker = sy.VirtualWorker(hook, id="secure_worker")

    with open("../configs/federated_train_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Load Bob's train/validation data
    print("Lodaing training/validation data ...")
    bob_dataset_train = TrafficDataset(config["dataset"]["train_dataset_bob"])
    bob_train_loader = DataLoader(bob_dataset_train, batch_size=config["train"]["batch_size"])

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    criterion = torch.nn.MSELoss()
    epochs = config["train"]["epochs"]
    for a_iter in range(epochs):
        model.send(bob)
        loss = 0
        for i, bobs_batch in enumerate(bob_train_loader):
            # Train Bob's Model
            bobs_data = bobs_batch.send(bob)
            optimizer.zero_grad()
            bobs_pred = model(bobs_data)
            bobs_loss = criterion(bobs_pred, bobs_data)
            bobs_loss.backward()

            optimizer.step()
            loss += bobs_loss.get().item()

        model.get()
        # dict_params = dict(params)
        # for name1, param1 in params:
        #     print(name1)
        # bobs_model.move(secure_worker)

        # with torch.no_grad():
        #     model.weight.set_(((bobs_model.weight.data) / 1).get())
        #     model.bias.set_(((bobs_model.bias.data) / 1).get())

        print("Bob:" + str(loss/len(bob_train_loader)))

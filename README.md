# threat_control_neural_network_pipeline

## Dataset
Download dataset for training/testing from [UCI repository](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT).
## Training

#### Configuration
You can find different training configurations in the ./configs folder.
#### Start normal training
Script to start training neural network:
```shell script
python3 ./train/train.py --config=configs/default_train_config.yml
```
#### Start federated training
In order train federated neural networks, it's necessary to install *PySyft* library. Script to start training federated neural network:
```shell script
python3 ./train_federated/train.py --config=configs/federated_train_config.yml
```

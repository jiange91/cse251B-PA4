import torch

ROOT_STATS_DIR = './experiment_data'

# Put your other constants here.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

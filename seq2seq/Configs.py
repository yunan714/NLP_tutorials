import torch

class configs():
    def __init__(self):
        self.BATCH_SIZE = 128
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
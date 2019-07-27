import torch

class configs():
    def __init__(self):
        self.BATCH_SIZE = 128
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LEARNING_RATE = 1e-3
        self.N_EPOCHES = 10
        self.CLIP = 1
        self.EN_EMB_DIM = 256
        self.GE_EMB_DIM = 256
        self.HID_DIM = 512
        self.N_LAYERS = 2
        self.EN_DROPOUT = 0.5
        self.GE_DROPOUT = 0.5
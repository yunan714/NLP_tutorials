import torch
class configs:
    def __init__(self):
        # 模型结构
        self.INPUT_DIM = 0
        self.OUTPUT_DIM = 0
        self.BATCH_SIZE = 0
        # 超参数
        self.LAERNING_RATE = 0
        self.RANDOM_SEED = 0
        self.GRAD_CLIP = 0

        # 配置信息
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DATA_PATH ="processed_data"
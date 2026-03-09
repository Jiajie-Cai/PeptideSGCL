class EarlyStopping:
    def __init__(self, patience=8, delta=0.001, verbose=False):
        """
        Args:
            patience (int): 忍耐轮数，若 val_loss 多轮未提升则早停
            delta (float): 最小改善值，低于该值视为未改善
            verbose (bool): 是否打印提示
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


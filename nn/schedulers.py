class StepLR:
    def __init__(self, optimizer, step_size=5, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma

    def step(self, epoch):
        if epoch > 0 and epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

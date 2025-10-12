import torch


class Scheduler:
    def __init__(self, n_timesteps: int = 1000, start: float = 0.00085, end: float = 0.0120):
        self.n_timesteps = n_timesteps
        self.betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps, dtype=torch.float64) ** 2
        self.alphas = torch.cumprod(1 - self.betas, 0)
        self.one_minus_alphas = 1 - self.alphas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas)

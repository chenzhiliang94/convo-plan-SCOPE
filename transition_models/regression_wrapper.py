from torch import nn
import torch

class RegressionWrapper(nn.Module):
    def __init__(self, model, embedding_size=1024):
        super(RegressionWrapper, self).__init__()
        self.add_module("model", model)
        self.input_mean = nn.Parameter(torch.zeros(embedding_size), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(embedding_size), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(embedding_size), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(embedding_size), requires_grad=False)
        self.use_residuals = nn.Parameter(torch.tensor(True, dtype=bool), requires_grad=False)

    def set_parameters(self, input_mean, input_std, output_mean, output_std, use_residuals):
        self.input_mean = nn.Parameter(input_mean, requires_grad=False)
        self.input_std = nn.Parameter(input_std, requires_grad=False)
        self.output_mean = nn.Parameter(output_mean, requires_grad=False)
        self.output_std = nn.Parameter(output_std, requires_grad=False)
        self.use_residuals = nn.Parameter(torch.tensor(use_residuals, dtype=bool), requires_grad=False)

    def scale_input(self, x):
        x = (x - self.input_mean) / self.input_std
        return x

    def scale_output(self, y):
        y = y * self.output_std + self.output_mean
        return y

    def forward(self, x):
        scaled_x = self.scale_input(x)
        y = self.model(scaled_x[:,None])[0][:,0,:]
        y = self.scale_output(y)
        if self.use_residuals:
            y = x + y
        return y

    def sample(self, x, samples_per_input=1):
        scaled_x = self.scale_input(x)
        y = self.model.sample(scaled_x, samples_per_input = samples_per_input)
        y = self.scale_output(y)
        if self.use_residuals:
            y = x + y
        return y
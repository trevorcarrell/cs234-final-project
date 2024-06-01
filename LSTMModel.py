import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        # Define layer dimensions and number of layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # Construct the initial hidden state and cell state
        initial_hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        initial_cell = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Forward pass through the LSTM layer
        out, (hn, cn) = self.lstm(x, (initial_hidden, initial_cell))

        # Capture the output of the last time step
        out = self.linear(out[:, -1, :])

        return out
    

    
import torch
import torch.nn as nn

class RecModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_recommendations):
        super(RecModel, self).__init__()
        # Define layer dimensions and number of layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        self.num_recommendations = num_recommendations

        # Define the input embedding and prediction mask
        self.embed = nn.Embedding(output_dim, hidden_dim)
        self.mask = torch.ones(output_dim)

        # Define the LSTM layer
        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_dim)

        # Define the output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softmax(dim=0)
        )

    def forward(self, x, prev_a_hat=None, prev_feedback=None):
        # Construct the initial hidden state and cell state
        hidden_state = torch.zeros(self.hidden_dim)
        cell_state = torch.zeros(self.hidden_dim)

        # Reset the mask if this is the first sub-episode
        if prev_a_hat is None:
            self.mask = torch.ones(self.output_dim)

        # Initialize the output tensors
        a_hats = torch.empty((self.num_timesteps), dtype=torch.long)  # Model predictions at each timestep
        feedbacks = torch.empty((self.num_timesteps), dtype=torch.long)  # Feedback at each timestep
        masked_probs = torch.empty((self.num_timesteps, self.output_dim))  # Masked probabilities at each timestep
        a_hat = torch.tensor(0) if prev_a_hat is None else prev_a_hat
        feedback = 0 if prev_feedback is None else prev_feedback
        for i in range(self.num_timesteps): 
            a_hat = torch.tensor(a_hat).long()
            embedding_input = self.embed(a_hat) * feedback if i > 0 else torch.zeros_like(self.embed(a_hat))
            hidden_state, cell_state = self.lstm_cell(embedding_input, (hidden_state, cell_state))

            # Put the hidden state through the output layer
            output = self.output_layer(hidden_state)
            masked_output = output * self.mask

            # Get the top num_recommendations items:
            top_items = torch.topk(masked_output, self.num_recommendations).indices

            # Get our prediction (a_hat), which is the top item in the intersection of the top items and the sub episode (x)
            intersection = top_items[torch.isin(top_items + 1, x)]
            intersection_cardinality = len(intersection)
            feedback = 1 if intersection_cardinality > 0 else -1
            a_hat = intersection[0] if intersection_cardinality > 0 else top_items[0]

            # Update the mask
            mask_clone = self.mask.clone()
            mask_clone[a_hat] = 0
            self.mask = mask_clone

            # Update a_hats, feedbacks, and masked_probs
            a_hats[i] = a_hat
            feedbacks[i] = feedback
            masked_probs[i] = masked_output

        return a_hats, feedbacks, masked_probs
    

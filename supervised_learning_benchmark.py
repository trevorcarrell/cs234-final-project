import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

PAD_VALUE = -100

def collate_fn(batch):
    """
    Custom collate function for the DataLoader that pads the sequences in each batch.
    """
    users = [item[0] for item in batch]
    episodes = [item[1] for item in batch]
    lens = [item[2] for item in batch]

    # Pad the sequences in each batch
    episodes = pad_sequence(episodes, batch_first=True, padding_value=PAD_VALUE)

    return torch.tensor(users), episodes, torch.tensor(lens)
    
def get_user_interests(df: pd.DataFrame, users: np.ndarray) -> dict[int, list[int]]:
    """
    Returns a dictionary of items tat each user has interacted with sorted by timestamp.
    """
    user_interests = defaultdict(list)

    # For each user, where users are numbered from 1 to n_users + 1
    items = []
    user_ids = []
    for user_id in users:
        user_ids.append(user_id)
        items.append(list(df[df['user'] == user_id]['item']))

    # Create a df with users and items
    user_interests = pd.concat([pd.Series(user_ids), pd.Series(items)], axis=1)
    user_interests.columns = ['user', 'items']

    return user_interests

def get_train_test_data(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Separates the data into training and testing sets, while also
    returning the number of unique users in each set.
    """
    unique_users = data_df.user.unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.1, random_state=429)
    train_data = data_df[data_df.user.isin(train_users)]
    test_data = data_df[data_df.user.isin(test_users)]

    return train_data, test_data, np.unique(train_users), np.unique(test_users)


def load_data(data_file):
    # Load the data in chunks and put it in a pandas dataframe:
    chunks = []

    for chunk in tqdm(pd.read_csv(data_file, header=None, sep=r'\s+', names=['user', 'item', 'rating', 'timestamp'], chunksize=1000)):
        chunks.append(chunk)
    
    # Concatenate the chunks into a single dataframe
    data = pd.concat(chunks, axis=0)

    # Sort users by timestamp
    data.sort_values('timestamp', inplace=True)

    return data


class SLDataset(Dataset):
    def __init__(self, data, num_timesteps):
        self.data = data
        self.num_timesteps = num_timesteps

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get a user and the items they like
        user, items = self.data.iloc[idx].values
        
        # Split the items into episodes of length num_timesteps
        episodes = []
        for i in range(0, len(items), self.num_timesteps):
            episodes.append(items[i : i + self.num_timesteps])
        
        # Exclude the last episode if it is not of length num_timesteps
        if len(episodes[-1]) != self.num_timesteps:
            episodes.pop()

        return torch.tensor(user), torch.tensor(episodes), torch.tensor(len(episodes))
        

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_recommendations):
        super(LSTMModel, self).__init__()
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
    

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Supervised Learning Benchmark')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the model')
    parser.add_argument('--input_dim', type=int, default=100, help='Input dimension of the model')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension of the model')
    parser.add_argument('--num_timesteps', type=int, default=20, help='Number of layers in the model')
    parser.add_argument('--num_recs', type=int, default=10, help='Number of recommendations to make')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training the model')
    parser.add_argument('--data_file', type=str, default='ml-100k/u.data', help='Path to the data file')
    parser.add_argument('--model_file', type=str, default='sl_lstm_model.pth', help='Path to save the model')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load a pre-trained model')

    return parser.parse_args()


def calculate_hit_at_n(feedbacks):
    num_users = len(feedbacks)

    # Calculate the hit@N, where N is the number of recommendations
    hit_at_n = 0.0
    for user_id, user_feedbacks in feedbacks.items():
        hit_at_n += (1 / len(user_feedbacks)) * torch.sum(user_feedbacks == 1)

    return hit_at_n / num_users


def test_model(model, test_loader, criterion):
    # Evalaute the trained model on the test set
    model.eval()
    test_loss = 0.0
    all_feedbacks = {}
    for i, data in enumerate(test_loader, 0):
        user, episodes, episode_len = data
        
        for e, episode in enumerate(episodes):
            user_feedbacks = []

            # Truncate the episode to the correct length
            episode = episode[:episode_len[e]]
            prev_a_hat = None
            prev_feedback = None

            for s, sub_episode in enumerate(episode):

                # Make a forward pass and calculate the loss
                if s == 0:  # First sub-episode
                    a_hats, feedbacks, masked_probs = model(sub_episode)
                else:  # Subsequent sub-episodes
                    a_hats, feedbacks, masked_probs = model(sub_episode, prev_a_hat, prev_feedback)

                # Store the previous a_hat and feedback for next sub-episode
                prev_a_hat = a_hats[-1]
                prev_feedback = feedbacks[-1]

                user_feedbacks.append(feedbacks)
                loss = criterion(masked_probs, sub_episode)
                test_loss += loss.item()
            
            all_feedbacks[user[e].item()] = torch.cat(user_feedbacks, dim=0)

    # Calculate the hit@N, where N is the number of recommendations
    hit_at_n = calculate_hit_at_n(all_feedbacks)
    print(f'Test Loss: {test_loss}')
    print(f'Hit@{model.num_recommendations}: {hit_at_n}')

    return hit_at_n, test_loss


def train_model(model, criterion, optimizer, train_loader, num_epochs, model_file):
    # Train the model for num_epochs
    model.train()
    all_feedbacks = {}
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            user, episodes, episode_len = data

            for e, episode in enumerate(episodes):
                user_feedbacks = []

                # Truncate the episode to the correct length
                episode = episode[:episode_len[e]]
                prev_a_hat = None
                prev_feedback = None

                for s, sub_episode in enumerate(episode):

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Make a forward pass and calculate the loss
                    if s == 0:  # First sub-episode
                        a_hats, feedbacks, masked_probs = model(sub_episode)
                    else:  # Subsequent sub-episodes
                        a_hats, feedbacks, masked_probs = model(sub_episode, prev_a_hat, prev_feedback)

                    # Store the previous a_hat and feedback for next sub-episode
                    prev_a_hat = a_hats[-1]
                    prev_feedback = feedbacks[-1]

                    # Calculate the loss and update the model parameters
                    loss = criterion(masked_probs, sub_episode)
                    loss.backward()
                    optimizer.step()
                    user_feedbacks.append(feedbacks)

                    # Update the epoch loss
                    epoch_loss += loss.item()

                all_feedbacks[user[e].item()] = torch.cat(user_feedbacks, dim=0)
        
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss }')
    
    # Calculate the hit@N, where N is the number of recommendations
    hit_at_n = calculate_hit_at_n(all_feedbacks)
    print(f'Hit@{model.num_recommendations}: {hit_at_n}')

    # Save the model
    torch.save(model.state_dict(), model_file)

    return hit_at_n


def main():
    # Set the random seed for reproducibility
    torch.manual_seed(429)

    # Parse the command line arguments
    args = parse_args()

    model_file = f'{args.model_file}_n{args.num_recs}'

    # Load the user data
    data_df = load_data(args.data_file)

    # Get the number of unique items
    n_items = data_df.item.nunique()

    # Separate the data into training and testing sets
    train_data, test_data, train_users, test_users = get_train_test_data(data_df)

    # Get the items that each user has interacted with, key: user_id, value: list of item ids
    user_interests_train = get_user_interests(train_data, train_users)
    user_interests_test = get_user_interests(test_data, test_users)
    
    # Create Dataset and DataLoader objects for training and testing
    train_dataset = SLDataset(user_interests_train, args.num_timesteps)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_dataset = SLDataset(user_interests_test, args.num_timesteps)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    # Set up hyperparameters
    lr = args.lr
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    num_recommendations = args.num_recs
    num_timesteps = args.num_timesteps
    output_dim = n_items
    num_epochs = args.num_epochs

    # Initialize the model, loss function, and optimizer
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_timesteps, num_recommendations)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    else:
        train_model(model, criterion, optimizer, train_loader, num_epochs, model_file)

    # Test the model
    test_model(model, test_loader, criterion)

           
if __name__ == '__main__':
    main()

    
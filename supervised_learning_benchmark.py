import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import argparse
from torch.utils.data import Dataset, DataLoader


def get_user_interests(df: pd.DataFrame, n_users: int) -> dict[int, list[int]]:
    """
    Returns a dictionary of items tat each user has interacted with sorted by timestamp.
    """
    user_interests = defaultdict(list)

    # For each user, where users are numbered from 1 to n_users + 1
    users = []
    items = []
    for uid in range(1, n_users + 1):
        users.append(uid)
        items.append(list(df[df['user'] == uid]['item']))

    # Create a df with users and items
    user_interests = pd.concat([pd.Series(users), pd.Series(items)], axis=1)
    user_interests.columns = ['user', 'items']

    return user_interests

def get_train_test_data(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Separates the data into training and testing sets, while also
    returning the number of unique users in each set.
    """
    unique_users = data_df.user.unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.1)
    train_data = data_df[data_df.user.isin(train_users)]
    test_data = data_df[data_df.user.isin(test_users)]
    n_train_users = train_data.user.nunique()
    n_test_users = test_data.user.nunique()

    return train_data, test_data, n_train_users, n_test_users


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
    def __init__(self, data, memory_length):
        self.data = data
        self.memory_length = memory_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get a user and the items they like
        user = self.data.iloc[idx].user
        items = self.data.iloc[idx].item
        
        # Split the items into episodes of length memory_length
        episodes = []
        for i in range(0, len(items), self.memory_length):
            episodes.append(items[i : i + self.memory_length])
        
        return user, episodes
        
        

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        # Define layer dimensions and number of layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_dim)

        # Define the output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.Softmax()
        )


    def forward(self, x):
        # Construct the initial hidden state and cell state
        
        hits = []# size of 20, records if the ith timestep is a hit (1) or not a hit (0), AKA feedback (everything the user likes)
        out = []
        for i in range(self.num_layers):
            # Get the hidden state and cell state
            hidden_state, cell_state = self.lstm_cell(x, (hidden_state, cell_state))
            # record hits

        # for ekin, give a_hat at each time step
        return out, hits
    

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Supervised Learning Benchmark')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the model')
    parser.add_argument('--input_dim', type=int, default=100, help='Input dimension of the model')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension of the model')
    parser.add_argument('--num_layers', type=int, default=20, help='Number of layers in the model')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training the model')
    parser.add_argument('--data_file', type=str, default='ml-100k/u.data', help='Path to the data file')
    parser.add_argument('--model_file', type=str, default='sl_lstm_model.pth', help='Path to save the model')
    parser.add_argument('--memory_length', type=int, default=20, help='How far back a time step should look in the past')

    return parser.parse_args()


def main():
    # Parse the command line arguments
    args = parse_args()

    # Get the data file and model file
    data_file = args.data_file
    model_file = args.model_file

    # Load the user data
    data_df = load_data(data_file)

    # Get the number of unique items
    n_items = data_df.item.nunique()

    # Separate the data into training and testing sets
    train_data, test_data, n_train_users, n_test_users = get_train_test_data(data_df)

    # Get the items that each user has interacted with, key: user_id, value: list of item ids
    user_interests = get_user_interests(data_df, n_train_users)
    
    # Create Dataset and DataLoader objects for training and testing
    train_dataset = SLDataset(user_interests, args.memory_length)

    # Set up hyperparameters
    lr = args.lr
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    rnn_size = args.rnn_size
    num_layers = args.num_layers
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    output_dim = n_items
    memory_length = args.memory_length

    # Initialize the model
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

           

if __name__ == '__main__':
    main()

    
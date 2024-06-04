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
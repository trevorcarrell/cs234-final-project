# %%
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import argparse
from torch.utils.data import Dataset, DataLoader

class PolicyGradient(object):
    def __init__(self, data, config, seed, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment
                config: class with hyperparameters
                logger: logger instance from the logging module
        """

        self.init_policy()
        pass
    
    # %%
    def init_policy(self):
        # TODO: Instantiate model here
        pass


    def update_policy(self, episode):
        # Every data instance is an input + label pair
        for j, sub_episode in enumerate(episode): 
            # sub_episode will be a list of 20 item ids

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            feedback, output = self.policy(sub_episode)
            # output are actions and probabilities
            
            values = []
            probs = []
            # Populate values and probabilities from each timestep
            # as part of loss computation
            for f, (_, p) in zip(feedback, output): 
                values.append(1 if f > 0 else -0.2)
                probs.append(p)

            # Discounted rewards
            discounted_rewards = []
            cumulative_reward = 0
            gamma = 0.99  # discount factor
            for reward in reversed(values):
                cumulative_reward = reward + gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            
            discounted_rewards = torch.tensor(discounted_rewards)
            # outputs are (probability dist over each timestep)
            
            # Compute the loss and its gradients
            # loss is 
            
            # Loss 
            loss = torch.sum()
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            # running_loss += loss.item()
        
    # %%
    def train(self, model, loader, device, criterion, optimizer, config): 
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            for j, sub_episode in enumerate(loader): 
                # sub_episode will be a list of 20 item ids

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                feedback, output = model(sub_episode)
                # outputs are (probability dist over each timestep)

                # Compute the loss and its gradients
                loss = criterion(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()

            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        return last_loss

    # run model on some new users 
    def run(self): 
        pass

def main():
    pass


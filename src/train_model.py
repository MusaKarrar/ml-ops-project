import os

import argparse
import torch

from models.model import *
from visualizations.visualize import *



parser = argparse.ArgumentParser(description="Script for training model")
parser.add_argument("--lr", default=1e-3, help="learning rate to use for training")
parser.add_argument("--epochs", default=5, help="epochs to train for")
parser.add_argument("--ckpt_name", default="ckpt_1.pth", help="Name of trained model")
parser.add_argument("--train_data_path", default="data/processed/train_images.pt", help="Path to training data")

def train(lr, epochs, ckpt_name, train_data_path):
    """
    Train model.
    args:
        lr: learning rate to use for training
        epochs: epochs to train for
        ckpt_name: Name of trained model
    
    """

    print("Training day and night")
    print(lr)
    print(epochs)
    print(ckpt_name)
    print(train_data_path)

    model = ConvNet2D()

    train_images = torch.load(train_data_path)
    train_targets = torch.load("data/processed/train_targets.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_targets)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    train_loss_epoch = []
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # add dim for conv2d
            images.resize_(images.shape[0], 1, 28, 28)
            output = model.forward(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Training loss: {running_loss/len(train_set)}")
        print(f"epoch: ", epoch + 1)

        train_loss_epoch.append(running_loss / len(train_set))

    for i in range(1000):
        save_path = os.path.join(rf"models/run_{i}", ckpt_name)
        fig_save_path = os.path.join(rf"reports/figures/run_{i}", ckpt_name.replace(".pth", ""))
        if not os.path.exists(save_path):
            os.makedirs(rf"models/run_{i}")
            os.makedirs(fig_save_path)
            torch.save(model.state_dict(), save_path)
            plot_train_loss_curve(train_loss_epoch, fig_save_path)

            break

    print("done training and saved model")


if __name__ == "__main__":

    args = parser.parse_args()
    print('args', args)

    for key, value in vars(args).items():
        globals()[key] = value

    train(lr, epochs, ckpt_name, train_data_path)

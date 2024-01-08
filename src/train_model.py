import os

import argparse
import torch
from tqdm import tqdm
from models.model import *
from visualizations.visualize import *

if torch.cuda.is_available():
    print("GPU is available.")
    gpu_available = True
else:
    print("GPU is not available. Switching to CPU.")

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

    #model = my_awesome_model()
    # Example usage:
    model = ViT()

    train_images = torch.load(train_data_path)
    train_targets = torch.load("data/processed/train_targets.pt")

    if gpu_available:
        model =  model.to('cuda')
        train_images = train_images.to('cuda')
        train_targets = train_targets.to('cuda')

    train_set = torch.utils.data.TensorDataset(train_images, train_targets)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    train_loss_epoch = []
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in tqdm(train_loader):
            # add dim for conv2dnet
            images.resize_(images.shape[0], 1, 28, 28) 
            #convert dtype of images to long
            images = images.float()
            output = model(images) # input does not have temporal structure/time dimension so we can just pass it as is. if we worked with text we would specify a mask.

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

    print(f"done training and saved model to {save_path}")


if __name__ == "__main__":

    args = parser.parse_args()
    print('args', args)

    for key, value in vars(args).items():
        globals()[key] = value

    train(lr, epochs, ckpt_name, train_data_path)

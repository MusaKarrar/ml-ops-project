import os

import argparse
import torch
from tqdm import tqdm
from models.model import *
from visualizations.visualize import *
from sklearn.model_selection import train_test_split
if torch.cuda.is_available():
    print("GPU is available.")
    gpu_available = True
else:
    print("GPU is not available. Switching to CPU.")
    gpu_available = False

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

    #model = ConvNet2D()
    model = ViT()

    data_images = torch.load(train_data_path)
    data_targets = torch.load("data/processed/train_targets.pt")

    if gpu_available:
        model =  model.to('cuda')
        data_images = data_images.to('cuda')
        data_targets = data_targets.to('cuda')

    # split data_images and data_targets into train and validation data
    train_images, val_images, train_targets, val_targets = train_test_split(data_images, data_targets, test_size=0.05, random_state=0)

    data_set = torch.utils.data.TensorDataset(train_images, train_targets)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    optimizer.zero_grad()
    train_loss_epoch = []
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in tqdm(train_loader):
            # add dim for conv2dnet
            #convert dtype of images to long
            images = images.float()
            output = model(images) # input does not have temporal structure/time dimension so we can just pass it as is. if we worked with text we would specify a mask.
            output = output.flatten()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Training loss (MSE): {running_loss/len(train_targets)}")
        #print validation loss
        model.eval()
        output = model(val_images) # input does not have temporal structure/time dimension so we can just pass it as is. if we worked with text we would specify a mask.
        output = output.flatten()
        val_loss = criterion(output, val_targets)

        val_loss /= len(val_targets)
        print(f'Validation loss (MSE):', val_loss)
        model.train()

        print(f"epoch: ", epoch + 1)

        train_loss_epoch.append(running_loss / len(train_targets))
    
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

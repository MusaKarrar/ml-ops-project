import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from models.model import *

parser = argparse.ArgumentParser(description="Script for training model")
parser.add_argument("--model_path", default="models/run_2/ckpt_1.pth", help="trained model_weights path")
parser.add_argument("--train_images", default="data/processed/train_images.pt", help="to visualize train features")



def plot_train_loss_curve(train_loss, fig_save_path):
    """Plot train loss curve and save it to fig_save_path.
    args: 
        train_loss: list of train loss
        fig_save_path: path to save the figure
    """
    plt.plot(train_loss)
    plt.savefig(os.path.join(fig_save_path, "train_loss.png"))
    

def get_activation(name):
    """Hook to get the activation of a (intermediate) layer. Used to visualize the feature maps. 
    args:
        name: name of the layer
    """
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def plot_conv_features(image_array, input_img, fig_save_path):
    """Plot the feature maps of the first convolutional layer.
    args:
        image_array: numpy array/tensor array of shape [64, 22, 22]
        input_img: numpy array/tensor of shape [28, 28]
        fig_save_path: path to save the figure
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Flatten the 2D array of subplots to simplify indexing
    axs = axs.flatten()

    # Loop through the subplots and display the images
    for i, ax in enumerate(axs):
        if i == 0:
            if len(input_img.shape) == 3:
                input_img = input_img[0]
            img = input_img
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Input data")
        else:
            img = image_array[i - 1, :, :]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Feature map {i}")

        ax.axis("off")  # Turn off axis labels for images

    # Adjust layout for better spacing
    plt.tight_layout()
    # Then save here:
    plt.savefig(os.path.join(fig_save_path, "conv2_features.png"))
    pass


def plot_TSNE_dim_reduction(train_images, train_targets, N, fig_save_path):
    """Plot the TSNE dimension reduction of the train images.
    args:
        train_images: numpy array/tensor array of shape [>=N, 28, 28]
        train_targets: numpy array/tensor array of shape [>=N]
        N: number of samples to plot
        fig_save_path: path to save the figure
    """
    TSNE_features = TSNE_model.fit_transform(train_images[:N])

    fig = plt.figure(figsize=(8, 8))

    for i in np.unique(train_targets[:N]):
        class_train_idx = np.where(train_targets[:N].numpy() == i)
        plt.scatter(TSNE_features[class_train_idx][:, 0], TSNE_features[class_train_idx][:, 1], label=i)
    plt.legend()
    plt.tight_layout()
    # Then save here:
    plt.savefig(os.path.join(fig_save_path, f"TSNE_reduction, N={N}.png"))
    pass


if __name__ == "__main__":

    args = parser.parse_args()

    fig_save_path = r"reports/figures/run_2/ckpt_1"

    train_images = torch.load("data/processed/train_images.pt")
    train_targets = torch.load("data/processed/train_targets.pt")

    model = ConvNet2D()
    criterion = torch.nn.CrossEntropyLoss()

    model_weights = torch.load(args.model_path)
    model.load_state_dict(model_weights)
    activation = {} # dictionary to store the activation of the conv2 layer - not sure if needed to state this
    model.conv2.register_forward_hook(get_activation("conv2"))

    input_data = train_images[0:64]  # just do first 64 of numpy array.

    conv2_output = model(input_data)
    feature_maps = activation["conv2"].detach()

    plot_conv_features(feature_maps[0], input_data[0], fig_save_path)

    TSNE_model = TSNE(n_components=2, random_state=0)

    N = 10000
    plot_TSNE_dim_reduction(train_images, train_targets, N, fig_save_path)
    print("done")

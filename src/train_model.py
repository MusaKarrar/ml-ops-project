import os
os.environ["WAND_API_KEY"] = "38c3d61662e5a11172dddf1df24561c66b8ed9cb"
import argparse
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from tqdm import tqdm
from models.model import *
from visualizations.visualize import *
from sklearn.model_selection import train_test_split
import wandb

sweep_config = {
    'name': 'sweep',
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize',
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.1,
        },
        'optimizer': {
            'values': ['sgd'],
        },
    },
}

# Initialize wandb
wandb.init(
    project="ml-ops-project"
)

sweep_id = wandb.sweep(sweep=sweep_config, project = "ml-ops-project")

if torch.cuda.is_available():
    print("GPU is available.")
    gpu_available = True
else:
    print("GPU is not available. Switching to CPU.")
    gpu_available = False
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
    # Log hyperparameters to wandb and set run name to checkpoint name
    wandb.config.update({"lr": lr, "epochs": epochs})  
    wandb.run.name = ckpt_name 

    print("Training day and night")
    print(lr)
    print(epochs)
    print(ckpt_name)
    print(train_data_path)

    #model = ConvNet2D()
    model = ViT()

    # Initialize watch to log grandients / parameters
    wandb.watch(model)

    data_images = torch.load(train_data_path)
    data_targets = torch.load("data/processed/train_targets.pt")

    if gpu_available:
        model =  model.to('cuda')
        data_images = data_images.to('cuda')
        data_targets = data_targets.to('cuda')
        data_images = data_images.to('cuda')
        data_targets = data_targets.to('cuda')

    # split data_images and data_targets into train and validation data
    train_images, val_images, train_targets, val_targets = train_test_split(data_images, data_targets, test_size=0.05, random_state=0)

    data_set = torch.utils.data.TensorDataset(train_images, train_targets)
    # split data_images and data_targets into train and validation data
    train_images, val_images, train_targets, val_targets = train_test_split(data_images, data_targets, test_size=0.05, random_state=0)

    data_set = torch.utils.data.TensorDataset(train_images, train_targets)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=16, shuffle=True)

    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    optimizer.zero_grad()
    train_loss_epoch = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
        for epoch in range(epochs):
            running_loss = 0
            for images, labels in tqdm(train_loader):
                # add dim for conv2dnet
                #convert dtype of images to long
                images = images.float()
                output = model(images) # input does not have temporal structure/time dimension so we can just pass it as is. if we worked with text we would specify a mask.
                output = output.flatten()
                output = output.flatten()
                loss = criterion(output, labels)
                wandb.log({"Training Loss": loss.item()}) # Log loss to wandb
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Training loss: {running_loss/len(train_targets)}")
            #print validation loss
            model.eval()
            output = model(val_images) # input does not have temporal structure/time dimension so we can just pass it as is. if we worked with text we would specify a mask.
            output = output.flatten()
            val_loss = criterion(output, val_targets)

            val_loss /= len(val_targets) 
            wandb.log({"val_loss": val_loss.item()}) # Log validation loss to wandb
            print(f'Validation loss:', val_loss)
            model.train()

            print(f"epoch: ", epoch + 1)

            train_loss_epoch.append(running_loss / len(train_targets))
        
    # Save model to wandb and finish wandb run
    wandb.save(os.path.join(wandb.run.dir, ckpt_name)) 
    wandb.finish() 
    
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

    # Save the profiler output, uncomment below if tensorboard is not used
    ## prof.export_chrome_trace("profiler_trace.json")


if __name__ == "__main__":

    args = parser.parse_args()
    print('args', args)

    for key, value in vars(args).items():
        globals()[key] = value

    train(lr, epochs, ckpt_name, train_data_path)

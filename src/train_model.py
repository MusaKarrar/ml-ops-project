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
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path = "config", config_name = "config_model.yaml")


def train(config):
    """ Train the ViT on our dataset"""
    print(f"Configuration: \n{OmegaConf.to_yaml(config)}")
    hparams = config.model_type
    torch.manual_seed(hparams["seed"]) 

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


def train(cfg):
    """
    Train model.
    args:
        lr: learning rate to use for training
        epochs: epochs to train for
        ckpt_name: Name of trained model
    
    """
    # Log hyperparameters to wandb and set run name to checkpoint name
    wandb.config.update({"lr": cfg.hyperparameters_ViT.lr if cfg.defaults.model_type == "ViT" else cfg.hyperparameters_CNN.lr, "epochs": cfg.hyperparameters_ViT.epochs if cfg.defaults.model_type == "ViT" else cfg.hyperparameters_CNN.epochs})  
    wandb.run.name = cfg.defaults.ckpt_name 

    print("Training day and night")
   

    #model = ConvNet2D()

    # Initialize watch to log grandients / parameters
    if cfg.defaults.model_type == "ViT":
        model = ViT(cfg.hyperparameters_ViT.img_shape,
               cfg.hyperparameters_ViT.in_channels,
               cfg.hyperparameters_ViT.patch_shape, #should be factor of 160 and 106
               cfg.hyperparameters_ViT.d_model,
               cfg.hyperparameters_ViT.num_transformer_layers, # from table 1 above
               cfg.hyperparameters_ViT.dropout_rate,
               cfg.hyperparameters_ViT.mlp_size,
               cfg.hyperparameters_ViT.num_heads,
               cfg.hyperparameters_ViT.num_classes #regression problem, so only one output

        )
    else:
        model =  ConvNet2D(eval(cfg.hyperparameters_CNN.img_shape), 
                     cfg.hyperparameters_CNN.in_channels, 
                     cfg.hyperparameters_CNN.conv_features_layer1, 
                     cfg.hyperparameters_CNN.conv_features_layer2, 
                     cfg.hyperparameters_CNN.kernel_size_layer1, 
                     cfg.hyperparameters_CNN.kernel_size_layer2, 
                     cfg.hyperparameters_CNN.maxpool_dim)    
    
    wandb.watch(model)

    data_images = torch.load(cfg.defaults.train_data_path)
    data_targets = torch.load("data/processed/train_targets.pt")

    if gpu_available:
        model =  model.to('cuda')
        data_images = data_images.to('cuda')
        data_targets = data_targets.to('cuda')
        data_images = data_images.to('cuda')
        data_targets = data_targets.to('cuda')

    # split data_images and data_targets into train and validation data
    train_images, val_images, train_targets, val_targets = train_test_split(data_images, data_targets, test_size=cfg.defaults.val_split, random_state=0)

    data_set = torch.utils.data.TensorDataset(train_images, train_targets)
    # split data_images and data_targets into train and validation data
    train_images, val_images, train_targets, val_targets = train_test_split(data_images, data_targets, test_size=0.05, random_state=0)

    data_set = torch.utils.data.TensorDataset(train_images, train_targets)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size= cfg.defaults.batch_size, shuffle=True)

    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.hyperparameters_ViT.lr if cfg.defaults.model_type == "ViT" else cfg.hyperparameters_CNN.lr)

    optimizer.zero_grad()
    train_loss_epoch = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
        for epoch in range(cfg.hyperparameters_ViT.epochs if cfg.defaults.model_type == "ViT" else cfg.hyperparameters_CNN.epochs):
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
    wandb.save(os.path.join(wandb.run.dir, cfg.defaults.ckpt_name)) 
    wandb.finish() 
    
    for i in range(1000):
        save_path = os.path.join(rf"models/run_{i}", cfg.defaults.ckpt_name)
        fig_save_path = os.path.join(rf"reports/figures/run_{i}", cfg.defaults.ckpt_name.replace(".pth", ""))
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
    cfg = OmegaConf.load("config/config_model.yaml")


    train(cfg)

from fastapi import FastAPI
from src.models.transformer import *
from src.models.convnet import *
import wandb
from omegaconf import OmegaConf

# Create FastAPI instance
# Initialize wandb

app = FastAPI()

def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
        model_weights: path to trained model_weights

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    return torch.cat([model(batch) for batch, _ in dataloader], 0)


@app.get("/")
 

async def root():
    wandb.init(
    project="ml-ops-project")
    wandb.run.name = "inference_run0_ckpt_1"
    cfg = OmegaConf.load("config/config_model.yaml")
    # load in model
    if cfg.defaults.model_type == "ViT":
        model = ViT(cfg.hyperparameters_ViT.img_shape,
                cfg.hyperparameters_ViT.in_channels,
                cfg.hyperparameters_ViT.patch_shape, #should be factor of 160 and 106
                cfg.hyperparameters_ViT.d_model,
                cfg.hyperparameters_ViT.num_transformer_layers, # from table 1 above
                cfg.hyperparameters_ViT.dropout_rate,
                cfg.hyperparameters_ViT.mlp_size,
                cfg.hyperparameters_ViT.num_heads,
                cfg.hyperparameters_ViT.num_classes) #regression problem, so only one output
    else:
        model = ConvNet2D(eval(cfg.hyperparameters_CNN.img_shape), 
                        cfg.hyperparameters_CNN.in_channels, 
                        cfg.hyperparameters_CNN.conv_features_layer1, 
                        cfg.hyperparameters_CNN.conv_features_layer2, 
                        cfg.hyperparameters_CNN.kernel_size_layer1, 
                        cfg.hyperparameters_CNN.kernel_size_layer2, 
                        cfg.hyperparameters_CNN.maxpool_dim)   
    model_weights = torch.load(f'models/run_{cfg.defaults.run_num_for_inference}/ckpt_1.pth')
    model.load_state_dict(model_weights)
    wandb.watch(model)

    test_tensor = torch.load(cfg.defaults.data_for_model_inference_path)
    if cfg.defaults.targets_for_model_inference_path is not None:
        test_targets = torch.load(cfg.defaults.targets_for_model_inference_path)

    placeholder_targets = torch.zeros((test_tensor.shape[0], 1)) if test_targets is None else test_targets

    testset = torch.utils.data.TensorDataset(test_tensor, placeholder_targets)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    predicted_data = predict(model, dataloader)
    wandb.log({"predicted_data": predicted_data})
    wandb.finish()
    predicted_data = predicted_data.tolist()
    test_targets = test_targets.tolist()
    pred_vs_true = list(zip(predicted_data, test_targets))
    return pred_vs_true if test_targets is not None else predicted_data
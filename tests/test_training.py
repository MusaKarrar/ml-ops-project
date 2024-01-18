from tests import _model_path, _test_images_path, _test_targets_path, _config_file
import torch
from src.models.transformer import ViT
from src.models.convnet import ConvNet2D
from omegaconf import OmegaConf

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



def test_loss_of_model_less():
    """
    Test that absolute mean loss of model is less than 200 for test data
    """
    cfg = OmegaConf.load(_config_file)
    #load model
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
    
    model.load_state_dict(torch.load(_model_path))
    model.eval()
    #predict
    test_images = torch.load(_test_images_path)
    test_targets = torch.load(_test_targets_path)

    testset = torch.utils.data.TensorDataset(test_images, test_targets)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    predicted_nitrogen = predict(model, dataloader)
    predicted_nitrogen = predicted_nitrogen.flatten()
    print(len(test_targets))
    mean_abs_error = sum(abs(test_targets - predicted_nitrogen)) / len(test_targets)

    assert mean_abs_error < 200, "Expected mean absolute error less than 200, got %s" % mean_abs_error
    








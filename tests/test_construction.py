from omegaconf import OmegaConf
from tests import _config_file

def test_model_construction():
    cfg = OmegaConf.load(_config_file)
    patch_shape = eval(cfg.hyperparameters_ViT.patch_shape)
    img_shape = eval(cfg.hyperparameters_ViT.img_shape)

    #check patch size is divisible by image of size 160x106
    assert img_shape[0] % patch_shape[0] == 0, "Patch size is not divisible by image size in 1. dimension"
    assert img_shape[1] % patch_shape[1] == 0, "Patch size is not divisible by image size in 2. dimension"
    assert cfg.defaults.model_type in ["ViT", "ConvNet2D"], "Model type not supported"

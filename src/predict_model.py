import argparse

import numpy as np
import torch

from src.models.model import ConvNet2D


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")

    parser.add_argument("--model_path", default="models/run_1/ckpt_1.pth", help="trained model_weights path")
    parser.add_argument(
        "--eval_data_path", default="data/processed/test_numpy.npy", help="files to predict using model"
    )

    args = parser.parse_args()

    model = my_awesome_model()
    criterion = torch.nn.CrossEntropyLoss()

    model_weights = torch.load(args.model_path)
    model.load_state_dict(model_weights)

    test_array = np.load(args.eval_data_path)

    placeholder_targets = torch.zeros((test_array.shape[0], 1))

    test_tensor = torch.from_numpy(test_array.astype(np.float32))

    testset = torch.utils.data.TensorDataset(test_tensor, placeholder_targets)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    predicted_data = predict(model, dataloader)

    print("predicted data shape is:", predicted_data.shape)

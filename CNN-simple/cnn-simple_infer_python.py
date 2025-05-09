"""Load saved FNO1d to TorchScript and run inference example."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def deploy(saved_model: str, device: str, batch_size: int = 1) -> torch.Tensor:
    """
    Load TorchScript CNN1d-simple and run inference to learn identity
    mapping and increment of 1.

    Parameters
    ----------
    saved_model : str
        location of FNO1d model saved to Torchscript
    device : str
        Torch device to run model on, 'cpu' or 'cuda'
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    output : torch.Tensor
        result of running inference on model with example Tensor input
    """

    # Generate synthetic data
    torch.manual_seed(0)
    X = torch.randn(batch_size, 1, 20, 20)
    Y = X + 1.0

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        output = model(X)

    elif device == "cuda":
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load(saved_model)
        input_tensor_gpu = X.to(torch.device("cuda"))
        output_gpu = model.forward(input_tensor_gpu)
        output = output_gpu.to(torch.device("cpu"))

    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    return output, X, Y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--filepath",
        help="Path to the file containing the PyTorch model",
        type=str,
        default=os.path.dirname(__file__),
    )
    parsed_args = parser.parse_args()
    filepath = parsed_args.filepath
    saved_model_file = os.path.join(filepath, "saved_cnn-simple_model_cpu.pt")

    device_to_run = "cpu"
    batch_size_to_run = 10

    with torch.no_grad():
        predicted, X, Y = deploy(saved_model_file, device_to_run, batch_size_to_run)

    print((torch.abs(predicted - Y)).mean().item())
    # Compute absolute error
    error = np.abs(predicted.squeeze().numpy() - Y.squeeze().numpy())

    # Total error
    # print(error)
    # print(pred_vals.shape, Y.shape)
    # print(pred_vals[0][0][0])
    # print(Y[0][0][0])
    # print(Y.squeeze().numpy()[0])

    print(error.mean())

    print((torch.abs(X - Y)).mean().item())
    # tol_sum = 0.2
    # # Check against tolerance
    # if total_error < tol_sum:

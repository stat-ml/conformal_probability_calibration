import torch


def get_device(verbose: bool = False) -> torch.device:
    """
    Determines the most appropriate torch device available.

    Checks for CUDA, then MPS (for Apple Silicon), and falls back to CPU.

    Args:
        verbose (bool): If True, prints the selected device to the console.

    Returns:
        torch.device: The selected torch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Using device: {device}")

    return device

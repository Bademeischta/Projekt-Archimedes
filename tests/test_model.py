import torch
from src.archimedes.model import TPN

def test_tpn_forward_pass():
    model = TPN()
    # Create a dummy input tensor with the correct shape (batch_size, channels, height, width)
    dummy_input = torch.randn(4, 22, 8, 8)

    policy_output, value_output = model(dummy_input)

    # Check policy head output shape
    assert policy_output.shape == (4, 4672)

    # Check value head output shape
    assert value_output.shape == (4, 1)

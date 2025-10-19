import torch
from src.archimedes.model import TPN, SAN
from torch_geometric.data import Data, Batch

def test_tpn_forward_pass():
    model = TPN()
    # Create a dummy input tensor with the correct shape (batch_size, channels, height, width)
    dummy_input = torch.randn(4, 22, 8, 8)

    policy_output, value_output = model(dummy_input)

    # Check policy head output shape
    assert policy_output.shape == (4, 4672)

    # Check value head output shape
    assert value_output.shape == (4, 1)

def test_san_forward_pass():
    model = SAN()

    # Create a batch of dummy graphs
    graph1 = Data(x=torch.randn(64, 16), edge_index=torch.randint(0, 64, (2, 100)))
    graph2 = Data(x=torch.randn(64, 16), edge_index=torch.randint(0, 64, (2, 120)))
    batch = Batch.from_data_list([graph1, graph2])

    G_out, P_out, pi_out, a_sfs_out = model(batch)

    # Check Goal head output shape
    assert G_out.shape == (2, 20)
    # Check Plan Embedding head output shape
    assert P_out.shape == (2, 5, 256)
    # Check Plan Policy head output shape
    assert pi_out.shape == (2, 5)
    # Check A-SFS head output shape
    assert a_sfs_out.shape == (2, 1)

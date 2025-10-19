import torch
import chess

from src.archimedes.rewards import calculate_sfs
from src.archimedes.model import TPN, SAN

def test_calculate_sfs():
    # Initialize dummy models
    tpn = TPN()
    san = SAN()

    # Create a board and a goal vector
    board = chess.Board()
    original_goal_vector = torch.randn(1, 20)

    # Run the calculation
    sfs = calculate_sfs(board, original_goal_vector, tpn, san)

    # Check that the output is a float
    assert isinstance(sfs, float)

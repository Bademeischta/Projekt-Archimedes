import chess
import torch

from src.archimedes.model import TPN, SAN, PlanToMoveMapper
from src.archimedes.search import ConceptualGraphSearch
from src.archimedes.representation import board_to_tensor
import src.archimedes.representation as representation_module

def test_mcts_search():
    # This test is primarily to ensure the MCTS loop runs without crashing.
    tpn = TPN()
    san = SAN()
    mapper = PlanToMoveMapper()
    search = ConceptualGraphSearch(tpn, san, mapper, num_simulations=4) # Small number for test

    board = chess.Board()
    result = search.search(board, temperature=0)

    assert "best_move" in result
    assert isinstance(result["best_move"], chess.Move)
    assert result["best_move"] in board.legal_moves

def test_tactical_override_batched():
    class MockTPN(TPN):
        def forward(self, x):
            # If it's a batch (more than 1 item), it's the override check
            if x.shape[0] > 1:
                # One of the moves in the batch leads to a winning position for the opponent
                winning_value = torch.tensor([[0.9]])
                other_values = torch.full((x.shape[0] - 1, 1), 0.1)
                values = torch.cat([winning_value, other_values], dim=0)
                return torch.randn(x.shape[0], 4672), values
            # Otherwise, it's the initial check. Return a bad score.
            return torch.randn(1, 4672), torch.tensor([[-0.6]])

    tpn = MockTPN()
    san = SAN()
    mapper = PlanToMoveMapper()
    search = ConceptualGraphSearch(tpn, san, mapper, tactical_override_threshold=-0.5)

    board = chess.Board()
    result = search.search(board)

    # Assert that the override was triggered.
    assert search.tactical_overrides == 1
    # Assert that the returned move is a legal move.
    assert result["best_move"] in board.legal_moves

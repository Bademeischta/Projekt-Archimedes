import chess
import torch

from src.archimedes.model import TPN, SAN, PlanToMoveMapper
from src.archimedes.search import ConceptualGraphSearch
from src.archimedes.representation import board_to_tensor
import src.archimedes.representation as representation_module

def test_conceptual_graph_search():
    # Initialize dummy models
    tpn = TPN()
    san = SAN()
    mapper = PlanToMoveMapper()

    # Initialize the search
    search = ConceptualGraphSearch(tpn, san, mapper)

    # Create a board
    board = chess.Board()

    # Run the search
    result = search.search(board)

    # Check the output dictionary
    assert isinstance(result, dict)
    assert "best_move" in result
    assert "a_sfs_prediction" in result
    assert "original_goal_vector" in result
    assert "board_after_plan" in result

    # Check that the move is valid
    move = result["best_move"]
    assert isinstance(move, chess.Move)
    assert move in board.legal_moves

def test_tactical_override():
    # This mock TPN will return a bad score for the initial position,
    # and a very good score for the opponent if they play a specific move.
    class MockTPN(TPN):
        def __init__(self):
            super().__init__()
            self.call_count = 0
        def forward(self, x):
            self.call_count += 1
            # First call is the initial check. Return a bad score to trigger the override search.
            if self.call_count == 1:
                return torch.randn(1, 4672), torch.tensor([[-0.6]])
            # Second call is inside the override search. Return a winning score for the opponent.
            return torch.randn(1, 4672), torch.tensor([[0.9]])

    tpn = MockTPN()
    san = SAN()
    mapper = PlanToMoveMapper()
    search = ConceptualGraphSearch(tpn, san, mapper, tactical_override_threshold=-0.5)

    board = chess.Board()

    # Prepare tensors for monkey-patching
    normal_tensor = board_to_tensor(board.copy())
    threat_tensor = normal_tensor.clone()
    threat_tensor[0, 0, :3] = torch.tensor([1., 1., 1.])

    call_count = 0
    def mock_b2t(b):
        nonlocal call_count
        call_count += 1
        # The 2nd call happens inside the check, on the first legal move for the opponent
        if call_count == 2:
            return threat_tensor
        return normal_tensor

    # Monkey-patch board_to_tensor
    original_b2t = representation_module.board_to_tensor
    representation_module.board_to_tensor = mock_b2t

    result = search.search(board)

    # Restore
    representation_module.board_to_tensor = original_b2t

    # Assert that the override was triggered and the correct move was returned
    assert search.tactical_overrides == 1
    # The override should return a legal move.
    assert result["best_move"] in board.legal_moves

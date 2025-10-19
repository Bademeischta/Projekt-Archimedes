import chess
import torch

from src.archimedes.model import TPN, SAN, PlanToMoveMapper
from src.archimedes.search import ConceptualGraphSearch

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

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
    move = search.search(board)

    # Check that the output is a valid move
    assert isinstance(move, chess.Move)
    assert move in board.legal_moves

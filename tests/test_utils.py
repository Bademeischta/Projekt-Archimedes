import chess
import pytest

from src.archimedes.utils import move_to_index, index_to_move

@pytest.mark.parametrize("move_uci, fen", [
    ("e2e4", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("g1f3", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
    ("a7a8q", "rnbqkb1r/p1p1p1p1/1p1p1pPp/7P/8/8/PPPP1P2/RNBQKBNR w KQkq - 0 11"),
    ("a7a8r", "rnbqkb1r/p1p1p1p1/1p1p1pPp/7P/8/8/PPPP1P2/RNBQKBNR w KQkq - 0 11"),
    ("h2h1n", "rnbqkbnr/pppp2pp/8/4p3/4P1P1/8/PPPP1P1P/RNBQKBNR b KQkq g3 0 3"),
])
def test_move_index_conversion(move_uci, fen):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)

    index = move_to_index(move)
    converted_move = index_to_move(index, board)

    assert move == converted_move

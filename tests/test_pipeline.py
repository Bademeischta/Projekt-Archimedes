import io
import chess
from src.archimedes.pipeline import pgn_parser

def test_pgn_parser():
    pgn_string = '1. e4 e5 {A comment.} *\n'
    pgn_handle = io.StringIO(pgn_string)

    parser = pgn_parser(pgn_handle)
    data = next(parser)
    tensor_board, graph_board, comment, move = data

    assert tensor_board.shape == (22, 8, 8)
    assert graph_board.num_nodes == 64
    assert comment == "A comment."
    assert move == chess.Move.from_uci("e7e5")

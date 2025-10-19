from src.archimedes.pipeline import pgn_parser

def test_pgn_parser():
    parser = pgn_parser("tests/sample.pgn")
    data = next(parser)
    tensor_board, graph_board, comment = data

    assert tensor_board.shape == (22, 8, 8)
    assert graph_board.num_nodes == 64
    assert comment == "This is a comment."

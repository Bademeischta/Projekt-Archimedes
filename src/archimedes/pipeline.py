import chess.pgn
from typing import Iterator, Tuple

import torch
from torch_geometric.data import Data

from .representation import board_to_tensor, board_to_graph


def pgn_parser(pgn_file: str) -> Iterator[Tuple[torch.Tensor, Data, str]]:
    """
    Parses a PGN file and yields the tensor and graph representations of each board state,
    along with the raw human comment.

    Args:
        pgn_file: Path to the PGN file.

    Yields:
        A tuple containing:
        - tensor_board: The tensor representation of the board.
        - graph_board: The graph representation of the board.
        - comment: The raw human comment for the board state.
    """
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            for node in game.mainline():
                board.push(node.move)
                comment = node.comment
                if comment:
                    tensor_board = board_to_tensor(board)
                    graph_board = board_to_graph(board)
                    yield tensor_board, graph_board, comment

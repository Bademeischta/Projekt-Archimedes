import chess.pgn
from typing import Iterator, Tuple, IO

import torch
from torch_geometric.data import Data

from .representation import board_to_tensor, board_to_graph


def pgn_parser(pgn_handle: IO) -> Iterator[Tuple[torch.Tensor, Data, str, chess.Move]]:
    """
    Parses a PGN file and yields the tensor and graph representations of each board state,
    along with the raw human comment and the move that led to the state.

    Args:
        pgn_handle: A file-like object for the PGN file.

    Yields:
        A tuple containing:
        - tensor_board: The tensor representation of the board.
        - graph_board: The graph representation of the board.
        - comment: The raw human comment for the board state.
        - move: The `chess.Move` object.
    """
    while True:
        game = chess.pgn.read_game(pgn_handle)
        if game is None:
            break

        board = game.board()
        for node in game.mainline():
            move = node.move
            board.push(move)
            comment = node.comment
            if comment:
                tensor_board = board_to_tensor(board)
                graph_board = board_to_graph(board)
                yield tensor_board, graph_board, comment, move

import chess
import torch
import pytest

from src.archimedes.representation import board_to_tensor, board_to_graph

def test_board_to_tensor_shape():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (22, 8, 8)

def test_board_to_graph_nodes():
    board = chess.Board()
    graph = board_to_graph(board)
    assert graph.num_nodes == 64

def test_board_to_graph_node_features():
    board = chess.Board()
    graph = board_to_graph(board)
    assert graph.x.shape[1] == 16

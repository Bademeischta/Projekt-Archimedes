import tempfile
from pathlib import Path
import torch
import sys
import chess

from src.archimedes.create_dataset import main
from src.archimedes.utils import move_to_index

def test_create_dataset_script_tpn_format():
    with tempfile.TemporaryDirectory() as tempdir:
        input_dir = Path(tempdir) / "input"
        output_dir = Path(tempdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create a dummy pgn file with a known result
        dummy_pgn_path = input_dir / "sample.pgn"
        with open(dummy_pgn_path, "w") as f:
            f.write('[Event "Sample Game"]\n')
            f.write('[Result "1-0"]\n')
            f.write('\n')
            f.write('1. e4 e5 2. Nf3 *\n')

        # Run the script
        original_argv = sys.argv
        sys.argv = [
            "src/archimedes/create_dataset.py",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--shard-size", "1"
        ]

        main()

        sys.argv = original_argv  # Restore

        # Verify the output
        shard_files = sorted(list(output_dir.glob("*.pt")))
        assert len(shard_files) == 3 # Three moves, shard size of 1

        # Check the first shard (position before 1. e4)
        shard1_data = torch.load(shard_files[0])
        assert len(shard1_data) == 1
        tensor_board, move_index, game_result = shard1_data[0]

        assert tensor_board.shape == (22, 8, 8)
        assert game_result == 1.0

        board = chess.Board()
        expected_move = chess.Move.from_uci("e2e4")
        expected_index = move_to_index(expected_move)
        assert move_index == expected_index

        # Check the second shard (position before 1... e5)
        shard2_data = torch.load(shard_files[1])
        assert len(shard2_data) == 1
        tensor_board, move_index, game_result = shard2_data[0]

        assert tensor_board.shape == (22, 8, 8)
        assert game_result == 1.0

        board.push(expected_move) # Play 1. e4
        expected_move_2 = chess.Move.from_uci("e7e5")
        expected_index_2 = move_to_index(expected_move_2)
        assert move_index == expected_index_2

        # Check the third shard (position before 2. Nf3)
        shard3_data = torch.load(shard_files[2])
        assert len(shard3_data) == 1
        tensor_board, move_index, game_result = shard3_data[0]

        assert tensor_board.shape == (22, 8, 8)
        assert game_result == 1.0

        board.push(expected_move_2) # Play 1... e5
        expected_move_3 = chess.Move.from_uci("g1f3")
        expected_index_3 = move_to_index(expected_move_3)
        assert move_index == expected_index_3

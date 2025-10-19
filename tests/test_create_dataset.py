import tempfile
import shutil
from pathlib import Path
import torch
import os

from src.archimedes.create_dataset import main

def test_create_dataset_script():
    with tempfile.TemporaryDirectory() as tempdir:
        input_dir = Path(tempdir) / "input"
        output_dir = Path(tempdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create a dummy pgn file
        dummy_pgn_path = input_dir / "sample.pgn"
        with open(dummy_pgn_path, "w") as f:
            f.write('[Event "Sample Game"]\n')
            f.write('[Site "?"]\n')
            f.write('[Date ""]\n')
            f.write('[Round "?"]\n')
            f.write('[White "?"]\n')
            f.write('[Black "?"]\n')
            f.write('[Result "*"]\n')
            f.write('\n')
            f.write('1. e4 e5 {A comment.} 2. Nf3 Nc6 {Another comment.} *\n')

        # Run the script
        args = [
            "src/archimedes/create_dataset.py",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--shard-size", "1"
        ]

        import sys
        original_argv = sys.argv
        sys.argv = args

        main()

        sys.argv = original_argv # Restore original arguments

        # Verify the output
        shard_files = sorted(list(output_dir.glob("*.pt")))
        assert len(shard_files) == 2 # Two comments, shard size of 1

        # Check the first shard
        shard1_data = torch.load(shard_files[0], weights_only=False)
        assert len(shard1_data) == 1
        tensor_board, graph_board, comment = shard1_data[0]
        assert tensor_board.shape == (22, 8, 8)
        assert graph_board.num_nodes == 64
        assert comment == "A comment."

        # Check the second shard
        shard2_data = torch.load(shard_files[1], weights_only=False)
        assert len(shard2_data) == 1
        tensor_board, graph_board, comment = shard2_data[0]
        assert tensor_board.shape == (22, 8, 8)
        assert graph_board.num_nodes == 64
        assert comment == "Another comment."

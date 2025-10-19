import tempfile
from pathlib import Path
import torch
import sys
import chess
import pytest

from src.archimedes.create_dataset import main
from src.archimedes.utils import move_to_index

@pytest.fixture
def setup_test_environment():
    with tempfile.TemporaryDirectory() as tempdir:
        input_dir = Path(tempdir) / "input"
        output_dir = Path(tempdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        yield input_dir, output_dir

def run_script(args):
    original_argv = sys.argv
    sys.argv = args
    main()
    sys.argv = original_argv

def test_create_dataset_tpn(setup_test_environment):
    input_dir, output_dir = setup_test_environment

    # Create a dummy pgn file
    (input_dir / "sample.pgn").write_text('[Result "1-0"]\n1. e4 e5 2. Nf3 *\n')

    # Run the script for TPN
    run_script([
        "src/archimedes/create_dataset.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--shard-size", "2",
        "--dataset-type", "tpn"
    ])

    # Verify the output
    shard_files = sorted(list(output_dir.glob("*.pt")))
    assert len(shard_files) == 2

    # Check first shard
    shard1_data = torch.load(shard_files[0])
    assert len(shard1_data) == 2
    _, _, game_result = shard1_data[0]
    assert game_result == 1.0

def test_create_dataset_san(setup_test_environment):
    input_dir, output_dir = setup_test_environment

    # Create a dummy pgn file with comments
    (input_dir / "sample.pgn").write_text('1. e4 {comment1} e5 {comment2} *\n')

    # Run the script for SAN
    run_script([
        "src/archimedes/create_dataset.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--shard-size", "1",
        "--dataset-type", "san"
    ])

    # Verify the output
    shard_files = sorted(list(output_dir.glob("*.pt")))
    assert len(shard_files) == 2

    # Check first shard
    shard1_data = torch.load(shard_files[0], weights_only=False)
    assert len(shard1_data) == 1
    graph_board, comment = shard1_data[0]
    assert graph_board.num_nodes == 64
    assert comment == "comment1"

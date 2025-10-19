import argparse
import os
from pathlib import Path
import torch
import chess.pgn

from .pipeline import pgn_parser
from .utils import move_to_index
from .representation import board_to_tensor

def get_game_result(game):
    result = game.headers.get("Result")
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Create a dataset from PGN files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing PGN files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the dataset shards.")
    parser.add_argument("--shard-size", type=int, default=10000, help="Number of positions per shard.")
    parser.add_argument("--dataset-type", type=str, required=True, choices=['tpn', 'san'], help="Type of dataset to create ('tpn' or 'san').")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pgn_files = list(Path(args.input_dir).glob("*.pgn"))
    print(f"Found {len(pgn_files)} PGN files to process.")

    shard_data = []
    shard_num = 1
    total_positions = 0

    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}...")
        try:
            with open(pgn_file) as f:
                if args.dataset_type == 'tpn':
                    while game := chess.pgn.read_game(f):
                        game_result = get_game_result(game)
                        board = game.board()
                        for node in game.mainline():
                            move = node.move
                            tensor_board = board_to_tensor(board.copy())
                            move_index = move_to_index(move)

                            shard_data.append((tensor_board, move_index, game_result))
                            total_positions += 1

                            if len(shard_data) >= args.shard_size:
                                shard_filename = output_path / f"shard_{shard_num}.pt"
                                print(f"Saving TPN shard {shard_num} with {len(shard_data)} positions...")
                                torch.save(shard_data, shard_filename)
                                shard_data = []
                                shard_num += 1

                            board.push(move)

                elif args.dataset_type == 'san':
                    for _, graph_board, comment, _ in pgn_parser(f):
                        shard_data.append((graph_board, comment))
                        total_positions += 1

                        if len(shard_data) >= args.shard_size:
                            shard_filename = output_path / f"shard_{shard_num}.pt"
                            print(f"Saving SAN shard {shard_num} with {len(shard_data)} positions...")
                            torch.save(shard_data, shard_filename)
                            shard_data = []
                            shard_num += 1

        except Exception as e:
            print(f"Error processing file {pgn_file}: {e}")

    if shard_data:
        shard_filename = output_path / f"shard_{shard_num}.pt"
        print(f"Saving final shard {shard_num} with {len(shard_data)} positions...")
        torch.save(shard_data, shard_filename)

    print(f"\nProcessing complete.")
    print(f"Total positions processed: {total_positions}")
    print(f"Total shards created: {shard_num - 1 if total_positions > 0 else 0}")

if __name__ == "__main__":
    from .representation import board_to_tensor
    main()

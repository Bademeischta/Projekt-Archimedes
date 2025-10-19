import argparse
import os
from pathlib import Path
import torch

from .pipeline import pgn_parser

def main():
    parser = argparse.ArgumentParser(description="Create a dataset from PGN files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing PGN files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the dataset shards.")
    parser.add_argument("--shard-size", type=int, default=10000, help="Number of positions per shard.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
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
            for tensor_board, graph_board, comment in pgn_parser(str(pgn_file)):
                shard_data.append((tensor_board, graph_board, comment))
                total_positions += 1

                if len(shard_data) >= args.shard_size:
                    shard_filename = output_path / f"shard_{shard_num}.pt"
                    print(f"Saving shard {shard_num} with {len(shard_data)} positions to {shard_filename}...")
                    torch.save(shard_data, shard_filename)
                    shard_data = []
                    shard_num += 1
        except Exception as e:
            print(f"Error processing file {pgn_file}: {e}")

    # Save the last shard if it's not empty
    if shard_data:
        shard_filename = output_path / f"shard_{shard_num}.pt"
        print(f"Saving final shard {shard_num} with {len(shard_data)} positions to {shard_filename}...")
        torch.save(shard_data, shard_filename)

    print(f"\nProcessing complete.")
    print(f"Total positions processed: {total_positions}")
    print(f"Total shards created: {shard_num if total_positions > 0 else 0}")


if __name__ == "__main__":
    main()

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse

from src.archimedes.model import TPN

def load_shards(shard_dir):
    all_tensors = []
    all_moves = []
    all_results = []

    for shard_file in Path(shard_dir).glob("*.pt"):
        print(f"Loading shard: {shard_file}")
        data = torch.load(shard_file)
        for tensor, move, result in data:
            all_tensors.append(tensor)
            all_moves.append(move)
            all_results.append(result)

    return (torch.stack(all_tensors),
            torch.tensor(all_moves, dtype=torch.long),
            torch.tensor(all_results, dtype=torch.float32).unsqueeze(1))


def main():
    parser = argparse.ArgumentParser(description="Train the TPN model.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing dataset shards.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    args = parser.parse_args()

    # Load data
    tensors, moves, results = load_shards(args.dataset_dir)
    dataset = TensorDataset(tensors, moves, results)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model, optimizer, loss
    model = TPN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_tensors, batch_moves, batch_results in loader:
            optimizer.zero_grad()

            policy_pred, value_pred = model(batch_tensors)

            policy_loss = policy_loss_fn(policy_pred, batch_moves)
            value_loss = value_loss_fn(value_pred, batch_results)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    main()

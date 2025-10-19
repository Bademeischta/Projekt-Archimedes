import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from pathlib import Path
import argparse

from src.archimedes.model import SAN

def heuristic_comment_parser(comment: str, G_dims=20):
    """
    A placeholder heuristic to generate a dummy goal vector from a comment.
    Example: if 'attack' is in the comment, set the 'king_attack' index to 1.0.
    """
    goal_vector = torch.zeros(G_dims)
    if 'attack' in comment.lower():
        goal_vector[0] = 1.0 # Index 0 represents 'king_attack'
    return goal_vector

def load_shards(shard_dir):
    all_graphs = []
    all_comments = []

    for shard_file in Path(shard_dir).glob("*.pt"):
        print(f"Loading shard: {shard_file}")
        # weights_only=False is required to load torch_geometric.data.Data objects
        data = torch.load(shard_file, weights_only=False)
        for graph, comment in data:
            all_graphs.append(graph)
            all_comments.append(comment)

    return all_graphs, all_comments

def main():
    parser = argparse.ArgumentParser(description="Train the SAN model.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing dataset shards.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    args = parser.parse_args()

    # Load data
    graphs, comments = load_shards(args.dataset_dir)
    # Generate dummy targets
    goal_targets = torch.stack([heuristic_comment_parser(c) for c in comments])
    policy_targets = torch.randint(0, 5, (len(graphs),)) # Random plan policy targets

    loader = DataLoader(list(zip(graphs, goal_targets, policy_targets)), batch_size=args.batch_size, shuffle=True)

    # Model, optimizer, loss
    model = SAN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    goal_loss_fn = torch.nn.BCELoss()
    policy_loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_graphs, batch_goals, batch_policies in loader:
            optimizer.zero_grad()

            goal_pred, _, policy_pred = model(batch_graphs)

            goal_loss = goal_loss_fn(goal_pred, batch_goals)
            policy_loss = policy_loss_fn(policy_pred, batch_policies)
            loss = goal_loss + policy_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    main()

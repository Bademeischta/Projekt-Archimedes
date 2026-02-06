import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import time

from src.archimedes.model import SAN
from src.archimedes.benchmark_utils import apply_auto_config, get_auto_config

def heuristic_comment_parser(comment: str, G_dims=20):
    """
    A placeholder heuristic to generate a dummy goal vector from a comment.
    Example: if 'attack' is in the comment, set the 'king_attack' index to 1.0.
    """
    goal_vector = torch.zeros(G_dims)
    if 'attack' in comment.lower():
        goal_vector[0] = 1.0 # Index 0 represents 'king_attack'
    return goal_vector

def load_shards(shard_dir, device):
    all_graphs = []
    all_comments = []

    shard_files = list(Path(shard_dir).glob("*.pt"))
    print(f"Gefundene Shards: {len(shard_files)}")
    
    for shard_file in tqdm(shard_files, desc="Lade Shards"):
        # weights_only=False is required to load torch_geometric.data.Data objects
        data = torch.load(shard_file, map_location=device, weights_only=False)
        for graph, comment in data:
            # Move graph data to device
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            if hasattr(graph, 'batch'):
                graph.batch = graph.batch.to(device)
            all_graphs.append(graph)
            all_comments.append(comment)

    return all_graphs, all_comments

def main():
    parser = argparse.ArgumentParser(description="Train the SAN model.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing dataset shards.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size. Auto-configured if --auto-config is used.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu). Auto-detects if not specified.")
    parser.add_argument("--auto-config", action="store_true", help="Use benchmark results for optimal configuration.")
    parser.add_argument("--benchmark-file", type=str, default="benchmark_results.json", help="Path to benchmark results file.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers. Auto-configured if --auto-config is used.")
    args = parser.parse_args()

    # Auto-Config anwenden
    if args.auto_config:
        print("🔧 Verwende automatische Konfiguration basierend auf Benchmark...")
        args = apply_auto_config(args, args.benchmark_file)
        auto_config = get_auto_config(args.device, args.benchmark_file)
        print(f"   Batch-Size: {args.batch_size}")
        print(f"   DataLoader Workers: {args.num_workers}")
    else:
        # Fallback-Werte wenn nicht gesetzt
        if args.batch_size is None:
            args.batch_size = 32
        if args.num_workers is None:
            args.num_workers = 2

    # GPU/Device Setup für Colab Local Runtime
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Verwende Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Speicher: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load data
    print("\nLade Daten...")
    start_time = time.time()
    graphs, comments = load_shards(args.dataset_dir, device)
    load_time = time.time() - start_time
    print(f"Daten geladen in {load_time:.2f} Sekunden")
    print(f"Anzahl Samples: {len(graphs)}")
    
    # Generate dummy targets
    print("Generiere Targets...")
    goal_targets = torch.stack([heuristic_comment_parser(c) for c in comments]).to(device)
    policy_targets = torch.randint(0, 5, (len(graphs),), device=device) # Random plan policy targets

    loader = DataLoader(
        list(zip(graphs, goal_targets, policy_targets)), 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    # Model, optimizer, loss
    model = SAN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    goal_loss_fn = torch.nn.BCELoss()
    policy_loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    print(f"\nStarte Training für {args.epochs} Epochen...")
    total_training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_goal_loss = 0
        total_policy_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoche {epoch+1}/{args.epochs}")
        for batch_idx, (batch_graphs, batch_goals, batch_policies) in enumerate(pbar):
            batch_start = time.time()
            
            optimizer.zero_grad()

            goal_pred, _, policy_pred, _ = model(batch_graphs) # Ignore A-SFS output for now

            goal_loss = goal_loss_fn(goal_pred, batch_goals)
            policy_loss = policy_loss_fn(policy_pred, batch_policies)
            loss = goal_loss + policy_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_goal_loss += goal_loss.item()
            total_policy_loss += policy_loss.item()
            
            batch_time = time.time() - batch_start
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Goal': f'{goal_loss.item():.4f}',
                'Policy': f'{policy_loss.item():.4f}',
                'Batch Zeit': f'{batch_time:.2f}s'
            })

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)
        avg_goal_loss = total_goal_loss / len(loader)
        avg_policy_loss = total_policy_loss / len(loader)
        
        print(f"\nEpoche {epoch+1}/{args.epochs} abgeschlossen:")
        print(f"  Durchschnittlicher Loss: {avg_loss:.4f}")
        print(f"  Goal Loss: {avg_goal_loss:.4f}")
        print(f"  Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Epochen-Zeit: {epoch_time:.2f} Sekunden ({epoch_time/60:.2f} Minuten)")
        
        if epoch < args.epochs - 1:
            remaining_epochs = args.epochs - epoch - 1
            estimated_remaining = epoch_time * remaining_epochs
            print(f"  Geschätzte verbleibende Zeit: {estimated_remaining:.2f} Sekunden ({estimated_remaining/60:.2f} Minuten)")
    
    total_training_time = time.time() - total_training_start
    print(f"\n=== Training abgeschlossen ===")
    print(f"Gesamte Trainingszeit: {total_training_time:.2f} Sekunden ({total_training_time/60:.2f} Minuten)")
    print(f"Durchschnittliche Zeit pro Epoche: {total_training_time/args.epochs:.2f} Sekunden")

if __name__ == "__main__":
    main()

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse
from tqdm import tqdm
import time

from src.archimedes.model import TPN
from src.archimedes.benchmark_utils import apply_auto_config, get_auto_config

def load_shards(shard_dir, device):
    all_tensors = []
    all_moves = []
    all_results = []

    shard_files = list(Path(shard_dir).glob("*.pt"))
    print(f"Gefundene Shards: {len(shard_files)}")
    
    for shard_file in tqdm(shard_files, desc="Lade Shards"):
        data = torch.load(shard_file, map_location=device)
        for tensor, move, result in data:
            all_tensors.append(tensor.to(device))
            all_moves.append(move)
            all_results.append(result)

    return (torch.stack(all_tensors),
            torch.tensor(all_moves, dtype=torch.long).to(device),
            torch.tensor(all_results, dtype=torch.float32).unsqueeze(1).to(device))


def main():
    parser = argparse.ArgumentParser(description="Train the TPN model.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing dataset shards.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size. Auto-configured if --auto-config is used.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu). Auto-detects if not specified.")
    parser.add_argument("--auto-config", action="store_true", help="Use benchmark results for optimal configuration.")
    parser.add_argument("--benchmark-file", type=str, default="benchmark_results.json", help="Path to benchmark results file.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers. Auto-configured if --auto-config is used.")
    parser.add_argument("--pin-memory", action="store_true", default=None, help="Use pin_memory for DataLoader. Auto-configured if --auto-config is used.")
    args = parser.parse_args()

    # Auto-Config anwenden
    if args.auto_config:
        print("🔧 Verwende automatische Konfiguration basierend auf Benchmark...")
        args = apply_auto_config(args, args.benchmark_file)
        auto_config = get_auto_config(args.device, args.benchmark_file)
        print(f"   Batch-Size: {args.batch_size}")
        print(f"   DataLoader Workers: {args.num_workers}")
        print(f"   Pin Memory: {args.pin_memory}")
    else:
        # Fallback-Werte wenn nicht gesetzt
        if args.batch_size is None:
            args.batch_size = 32
        if args.num_workers is None:
            args.num_workers = 2
        if args.pin_memory is None:
            args.pin_memory = False

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
    tensors, moves, results = load_shards(args.dataset_dir, device)
    load_time = time.time() - start_time
    print(f"Daten geladen in {load_time:.2f} Sekunden")
    print(f"Anzahl Samples: {len(tensors)}")
    
    dataset = TensorDataset(tensors, moves, results)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    # Model, optimizer, loss
    model = TPN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    # Training loop
    print(f"\nStarte Training für {args.epochs} Epochen...")
    total_training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoche {epoch+1}/{args.epochs}")
        for batch_idx, (batch_tensors, batch_moves, batch_results) in enumerate(pbar):
            batch_start = time.time()
            
            optimizer.zero_grad()

            policy_pred, value_pred = model(batch_tensors)

            policy_loss = policy_loss_fn(policy_pred, batch_moves)
            value_loss = value_loss_fn(value_pred, batch_results)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            batch_time = time.time() - batch_start
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Policy': f'{policy_loss.item():.4f}',
                'Value': f'{value_loss.item():.4f}',
                'Batch Zeit': f'{batch_time:.2f}s'
            })

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)
        avg_policy_loss = total_policy_loss / len(loader)
        avg_value_loss = total_value_loss / len(loader)
        
        print(f"\nEpoche {epoch+1}/{args.epochs} abgeschlossen:")
        print(f"  Durchschnittlicher Loss: {avg_loss:.4f}")
        print(f"  Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Value Loss: {avg_value_loss:.4f}")
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

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import chess
import argparse
from collections import deque
import random
import wandb
import os
from tqdm import tqdm
import time

from src.archimedes.model import TPN, SAN, PlanToMoveMapper
from src.archimedes.search import ConceptualGraphSearch
from src.archimedes.rewards import calculate_sfs
from src.archimedes.representation import board_to_tensor
from src.archimedes.utils import move_to_index
from src.archimedes.benchmark_utils import apply_auto_config, get_auto_config

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def self_play_worker(rank, world_size, model, replay_queue, num_games, device, use_ddp=False):
    if use_ddp:
        setup(rank, world_size)
    tpn, san, mapper = model
    # Unwrap DDP if needed
    if use_ddp:
        tpn = tpn.module if hasattr(tpn, 'module') else tpn
        san = san.module if hasattr(san, 'module') else san
        mapper = mapper.module if hasattr(mapper, 'module') else mapper
    # Move models to device
    tpn = tpn.to(device)
    san = san.to(device)
    mapper = mapper.to(device)
    search = ConceptualGraphSearch(tpn, san, mapper)

    pbar = tqdm(range(num_games), desc=f"Self-Play Worker {rank}", position=rank, leave=False)
    for game_idx in pbar:
        game_start = time.time()
        board = chess.Board()
        game_history = []
        move_count = 0
        while not board.is_game_over(claim_draw=True):
            search_result = search.search(board)
            game_history.append((board.fen(), search_result["best_move"]))
            board.push(search_result["best_move"])
            move_count += 1

        result = board.result(claim_draw=True)
        final_game_result = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
        game_time = time.time() - game_start

        if rank == 0 or not use_ddp:
            replay_queue.put((game_history, final_game_result))
        
        pbar.set_postfix({
            'Moves': move_count,
            'Zeit': f'{game_time:.2f}s',
            'Ergebnis': result
        })

    if use_ddp:
        cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_step(tpn, san, mapper, replay_buffer, batch_size, tpn_optimizer, san_optimizer, device):
    experiences = replay_buffer.sample(batch_size)

    # Unwrap DDP models if needed for forward pass
    tpn_model = tpn.module if hasattr(tpn, 'module') else tpn
    san_model = san.module if hasattr(san, 'module') else san
    mapper_model = mapper.module if hasattr(mapper, 'module') else mapper

    board_tensors = torch.stack([exp["board_tensor"] for exp in experiences]).to(device)
    final_policies = torch.cat([exp["final_policy"] for exp in experiences]).to(device)
    v_tacticals = torch.cat([exp["v_tactical"] for exp in experiences]).to(device)
    a_sfs_predictions = torch.cat([exp["a_sfs_prediction"] for exp in experiences]).to(device)
    game_results = torch.tensor([exp["final_game_result"] for exp in experiences], device=device).unsqueeze(1)
    best_move_indices = torch.tensor([exp["best_move_index"] for exp in experiences], device=device)
    plan_policies = torch.cat([exp["plan_policy"] for exp in experiences]).to(device)

    real_sfs_list = [calculate_sfs(exp["board_after_plan"], exp["original_goal_vector"], tpn_model, san_model) for exp in experiences]
    real_sfs = torch.tensor(real_sfs_list, dtype=torch.float32, device=device).unsqueeze(1)

    # TPN Loss
    tpn_optimizer.zero_grad()
    policy_loss = torch.nn.CrossEntropyLoss()(final_policies, best_move_indices)
    value_loss = torch.nn.MSELoss()(v_tacticals, game_results)
    tpn_loss = policy_loss + value_loss
    tpn_loss.backward(retain_graph=True)
    tpn_optimizer.step()

    # SAN Loss
    san_optimizer.zero_grad()
    a_sfs_loss = torch.nn.MSELoss()(a_sfs_predictions, real_sfs)
    advantages = real_sfs - a_sfs_predictions.detach()
    plan_policy_loss = -torch.mean(torch.log_softmax(plan_policies, dim=1) * advantages)
    san_loss = a_sfs_loss + plan_policy_loss
    san_loss.backward()
    san_optimizer.step()

    wandb.log({
        "tpn_loss": tpn_loss.item(), "san_loss": san_loss.item(),
        "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
        "a_sfs_loss": a_sfs_loss.item(), "plan_policy_loss": plan_policy_loss.item()
    })
    
    return {
        "tpn_loss": tpn_loss.item(),
        "san_loss": san_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "a_sfs_loss": a_sfs_loss.item(),
        "plan_policy_loss": plan_policy_loss.item()
    }

def main():
    parser = argparse.ArgumentParser(description="End-to-end training for Archimedes.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of self-play workers. Auto-configured if --auto-config is used.")
    parser.add_argument("--total-games", type=int, default=10, help="Total number of self-play games to generate.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu). Auto-detects if not specified.")
    parser.add_argument("--training-iterations", type=int, default=100, help="Number of training iterations.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size. Auto-configured if --auto-config is used.")
    parser.add_argument("--auto-config", action="store_true", help="Use benchmark results for optimal configuration.")
    parser.add_argument("--benchmark-file", type=str, default="benchmark_results.json", help="Path to benchmark results file.")
    parser.add_argument("--replay-buffer-size", type=int, default=None, help="Replay buffer size. Auto-configured if --auto-config is used.")
    args = parser.parse_args()

    # Auto-Config anwenden
    if args.auto_config:
        print("🔧 Verwende automatische Konfiguration basierend auf Benchmark...")
        args = apply_auto_config(args, args.benchmark_file)
        auto_config = get_auto_config(args.device, args.benchmark_file)
        print(f"   Self-Play Workers: {args.num_workers}")
        print(f"   Batch-Size: {args.batch_size}")
        print(f"   Replay Buffer Größe: {args.replay_buffer_size}")
    else:
        # Fallback-Werte wenn nicht gesetzt
        if args.num_workers is None:
            args.num_workers = 2
        if args.batch_size is None:
            args.batch_size = 32
        if args.replay_buffer_size is None:
            args.replay_buffer_size = 10000

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

    wandb.init(project="archimedes", config=args)

    # Initialize models on device
    tpn = TPN().to(device)
    san = SAN().to(device)
    mapper = PlanToMoveMapper().to(device)
    
    # Only use DDP if multiple GPUs are available
    use_ddp = torch.cuda.device_count() > 1 and args.num_workers > 1
    if use_ddp:
        print(f"Verwende {torch.cuda.device_count()} GPUs mit DDP")
        tpn = DDP(tpn)
        san = DDP(san)
        mapper = DDP(mapper)
    else:
        print("Verwende Single GPU/CPU (kein DDP)")

    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    replay_queue = mp.Queue()

    # Get actual model parameters (unwrap DDP if needed)
    tpn_params = tpn.module.parameters() if hasattr(tpn, 'module') else tpn.parameters()
    san_params = san.module.parameters() if hasattr(san, 'module') else san.parameters()
    mapper_params = mapper.module.parameters() if hasattr(mapper, 'module') else mapper.parameters()
    
    tpn_optimizer = torch.optim.Adam(tpn_params)
    san_optimizer = torch.optim.Adam(list(san_params) + list(mapper_params))

    # Self-play phase
    print(f"\nStarte Self-Play Phase mit {args.num_workers} Workern für {args.total_games} Spiele...")
    self_play_start = time.time()
    
    if args.num_workers > 1 and use_ddp:
        world_size = args.num_workers
        mp.spawn(self_play_worker,
                 args=(world_size, (tpn, san, mapper), replay_queue, args.total_games // world_size, device, use_ddp),
                 nprocs=world_size,
                 join=True)
    else:
        # Single worker mode (better for Colab local runtime)
        print("Verwende Single-Worker Modus (optimal für Colab Local Runtime)")
        self_play_worker(0, 1, (tpn, san, mapper), replay_queue, args.total_games, device, False)
    
    self_play_time = time.time() - self_play_start
    print(f"Self-Play Phase abgeschlossen in {self_play_time:.2f} Sekunden ({self_play_time/60:.2f} Minuten)")

    # Process games and train
    print(f"\nVerarbeite Spiele und starte Training...")
    training_start = time.time()
    games_processed = 0
    
    pbar = tqdm(total=args.training_iterations, desc="Training Iterationen")
    training_iteration = 0
    
    while training_iteration < args.training_iterations:
        # Process games from queue
        games_processed_this_iter = 0
        while not replay_queue.empty() and games_processed_this_iter < 10:
            game_history, final_game_result = replay_queue.get()
            board = chess.Board()
            for fen, move in game_history:
                board.set_fen(fen)
                # Unwrap DDP models if needed
                tpn_model = tpn.module if hasattr(tpn, 'module') else tpn
                san_model = san.module if hasattr(san, 'module') else san
                mapper_model = mapper.module if hasattr(mapper, 'module') else mapper
                search_result = ConceptualGraphSearch(tpn_model, san_model, mapper_model).search(board)

                experience = {
                    "board_tensor": board_to_tensor(board).to(device),
                    "final_policy": search_result["final_policy"].to(device),
                    "v_tactical": search_result["v_tactical"].to(device),
                    "a_sfs_prediction": search_result["a_sfs_prediction"].to(device),
                    "original_goal_vector": search_result["original_goal_vector"].to(device),
                    "board_after_plan": search_result["board_after_plan"],
                    "best_move_index": move_to_index(move),
                    "plan_policy": search_result["plan_policy"].to(device),
                    "final_game_result": final_game_result
                }
                replay_buffer.push(experience)
                final_game_result *= -1
            games_processed += 1
            games_processed_this_iter += 1

        # Training step
        if len(replay_buffer) > 32:
            iter_start = time.time()
            loss_dict = train_step(tpn, san, mapper, replay_buffer, 32, tpn_optimizer, san_optimizer, device)
            iter_time = time.time() - iter_start
            
            training_iteration += 1
            pbar.update(1)
            pbar.set_postfix({
                'TPN Loss': f'{loss_dict["tpn_loss"]:.4f}',
                'SAN Loss': f'{loss_dict["san_loss"]:.4f}',
                'Iter Zeit': f'{iter_time:.2f}s',
                'Buffer': f'{len(replay_buffer)}/{args.replay_buffer_size}'
            })
            
            if training_iteration < args.training_iterations:
                remaining_iters = args.training_iterations - training_iteration
                estimated_remaining = iter_time * remaining_iters
                if training_iteration % 10 == 0:
                    print(f"\nIteration {training_iteration}/{args.training_iterations}")
                    print(f"  Geschätzte verbleibende Zeit: {estimated_remaining:.2f} Sekunden ({estimated_remaining/60:.2f} Minuten)")
    
    pbar.close()
    total_training_time = time.time() - training_start
    print(f"\n=== Training abgeschlossen ===")
    print(f"Gesamte Trainingszeit: {total_training_time:.2f} Sekunden ({total_training_time/60:.2f} Minuten)")
    print(f"Spiele verarbeitet: {games_processed}")
    print(f"Training Iterationen: {training_iteration}")

if __name__ == "__main__":
    main()

"""
End-to-end training for Archimedes with resumable checkpoints, MetricsLogger,
warmup phase, thermal throttling, AMP (Automatic Mixed Precision), and advanced schedulers.
"""

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import chess
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.archimedes.model import TPN, SAN, PlanToMoveMapper
from src.archimedes.search import ConceptualGraphSearch
from src.archimedes.rewards import calculate_sfs
from src.archimedes.representation import board_to_tensor
from src.archimedes.utils import move_to_index
from src.archimedes.benchmark_utils import apply_auto_config, get_auto_config
from src.archimedes.hardware_utils import get_hardware_snapshot, get_gpu_temp_c, check_thermal_throttle

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from metrics import MetricsLogger


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
    if use_ddp:
        tpn = tpn.module if hasattr(tpn, 'module') else tpn
        san = san.module if hasattr(san, 'module') else san
        mapper = mapper.module if hasattr(mapper, 'module') else mapper
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
            search_result = search.search(board, temperature=1.0, add_dirichlet_noise=True)
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


def warmup_worker(replay_queue, num_games, device):
    """Fill replay buffer with random/heuristic games (no training)."""
    from src.archimedes.representation import board_to_tensor
    tpn = TPN().to(device)
    san = SAN().to(device)
    mapper = PlanToMoveMapper().to(device)
    search = ConceptualGraphSearch(tpn, san, mapper)
    for _ in tqdm(range(num_games), desc="Warmup"):
        board = chess.Board()
        game_history = []
        while not board.is_game_over(claim_draw=True):
            search_result = search.search(board, temperature=1.5)
            game_history.append((board.fen(), search_result["best_move"]))
            board.push(search_result["best_move"])
        result = board.result(claim_draw=True)
        final = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
        replay_queue.put((game_history, final))


def drain_replay_queue(replay_queue, replay_buffer, tpn, san, mapper, device, max_drain=None):
    """
    Robust drain mechanism: Process all games in the queue and add to replay buffer.
    Returns number of games drained.
    """
    games_drained = 0
    tpn_m = tpn.module if hasattr(tpn, 'module') else tpn
    san_m = san.module if hasattr(san, 'module') else san
    mapper_m = mapper.module if hasattr(mapper, 'module') else mapper
    search = ConceptualGraphSearch(tpn_m, san_m, mapper_m)
    
    while not replay_queue.empty():
        if max_drain is not None and games_drained >= max_drain:
            break
        try:
            game_history, final_game_result = replay_queue.get_nowait()
        except Exception:
            break
        
        board = chess.Board()
        # FIXED: Use separate variable to avoid shadowing
        current_result = final_game_result
        for fen, move in game_history:
            board.set_fen(fen)
            search_result = search.search(board)
            experience = {
                "board_tensor": board_to_tensor(board).to(device),
                "final_policy": search_result["final_policy"].to(device),
                "v_tactical": search_result["v_tactical"].to(device),
                "a_sfs_prediction": search_result["a_sfs_prediction"].to(device),
                "original_goal_vector": search_result["original_goal_vector"].to(device),
                "board_after_plan": search_result["board_after_plan"],
                "best_move_index": move_to_index(move),
                "plan_policy": search_result["plan_policy"].to(device),
                "final_game_result": current_result
            }
            replay_buffer.push(experience)
            current_result *= -1
        games_drained += 1
    
    return games_drained


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_step(tpn, san, mapper, replay_buffer, batch_size, tpn_optimizer, san_optimizer, device, scaler=None):
    """
    Training step with optional AMP (Automatic Mixed Precision).
    """
    experiences = replay_buffer.sample(batch_size)

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

    use_amp = scaler is not None and device.type == 'cuda'

    # TPN Training with AMP
    tpn_optimizer.zero_grad()
    if use_amp:
        with autocast():
            policy_loss = torch.nn.CrossEntropyLoss()(final_policies, best_move_indices)
            value_loss = torch.nn.MSELoss()(v_tacticals, game_results)
            tpn_loss = policy_loss + value_loss
        scaler.scale(tpn_loss).backward(retain_graph=True)
        scaler.unscale_(tpn_optimizer)
        grad_norm_tpn = torch.nn.utils.clip_grad_norm_(list(tpn_params_from(tpn)), float('inf'))
        scaler.step(tpn_optimizer)
    else:
        policy_loss = torch.nn.CrossEntropyLoss()(final_policies, best_move_indices)
        value_loss = torch.nn.MSELoss()(v_tacticals, game_results)
        tpn_loss = policy_loss + value_loss
        tpn_loss.backward(retain_graph=True)
        grad_norm_tpn = torch.nn.utils.clip_grad_norm_(list(tpn_params_from(tpn)), float('inf'))
        tpn_optimizer.step()

    # SAN Training with AMP
    san_optimizer.zero_grad()
    if use_amp:
        with autocast():
            a_sfs_loss = torch.nn.MSELoss()(a_sfs_predictions, real_sfs)
            advantages = real_sfs - a_sfs_predictions.detach()
            plan_policy_loss = -torch.mean(torch.log_softmax(plan_policies, dim=1) * advantages)
            san_loss = a_sfs_loss + plan_policy_loss
        scaler.scale(san_loss).backward()
        scaler.unscale_(san_optimizer)
        grad_norm_san = torch.nn.utils.clip_grad_norm_(list(san_params_from(san)) + list(mapper_params_from(mapper)), float('inf'))
        scaler.step(san_optimizer)
        scaler.update()
    else:
        a_sfs_loss = torch.nn.MSELoss()(a_sfs_predictions, real_sfs)
        advantages = real_sfs - a_sfs_predictions.detach()
        plan_policy_loss = -torch.mean(torch.log_softmax(plan_policies, dim=1) * advantages)
        san_loss = a_sfs_loss + plan_policy_loss
        san_loss.backward()
        grad_norm_san = torch.nn.utils.clip_grad_norm_(list(san_params_from(san)) + list(mapper_params_from(mapper)), float('inf'))
        san_optimizer.step()

    with torch.no_grad():
        _, pred_top = final_policies.max(1)
        top1 = (pred_top == best_move_indices).float().mean().item()
        top5 = (final_policies.argsort(dim=1, descending=True)[:, :5] == best_move_indices.unsqueeze(1)).any(1).float().mean().item()

    grad_norm = (float(grad_norm_tpn) + float(grad_norm_san)) / 2.0

    if WANDB_AVAILABLE:
        wandb.log({
            "tpn_loss": tpn_loss.item(), "san_loss": san_loss.item(),
            "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
            "a_sfs_loss": a_sfs_loss.item(), "plan_policy_loss": plan_policy_loss.item(),
            "top1_accuracy": top1, "top5_accuracy": top5, "grad_norm": grad_norm
        })

    return {
        "tpn_loss": tpn_loss.item(), "san_loss": san_loss.item(),
        "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
        "a_sfs_loss": a_sfs_loss.item(), "plan_policy_loss": plan_policy_loss.item(),
        "top1_accuracy": top1, "top5_accuracy": top5, "grad_norm": grad_norm,
    }


def tpn_params_from(tpn):
    return tpn.module.parameters() if hasattr(tpn, 'module') else tpn.parameters()


def san_params_from(san):
    return san.module.parameters() if hasattr(san, 'module') else san.parameters()


def mapper_params_from(mapper):
    return mapper.module.parameters() if hasattr(mapper, 'module') else mapper.parameters()


def save_checkpoint_atomic(checkpoint_dir, state, filename="latest_checkpoint.pt"):
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    final_path = path / filename
    tmp_path = path / (filename + ".tmp")
    torch.save(state, tmp_path)
    os.replace(str(tmp_path), str(final_path))


def load_checkpoint(checkpoint_dir, device, filename="latest_checkpoint.pt"):
    """
    FIXED: Use weights_only=True to prevent arbitrary code execution
    """
    path = Path(checkpoint_dir) / filename
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=True)


def save_replay_buffer(buffer, checkpoint_dir, filename="replay_buffer.pt"):
    path = Path(checkpoint_dir) / filename
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state = list(buffer.buffer)
    torch.save(state, path)


def load_replay_buffer_into(buffer, checkpoint_dir, filename="replay_buffer.pt"):
    """
    FIXED: Use weights_only=True to prevent arbitrary code execution
    """
    path = Path(checkpoint_dir) / filename
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    cap = buffer.buffer.maxlen
    buffer.buffer = deque(state, maxlen=cap)


def main():
    parser = argparse.ArgumentParser(description="End-to-end training for Archimedes.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--total-games", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--training-iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--auto-config", action="store_true")
    parser.add_argument("--benchmark-file", type=str, default="benchmark_results.json")
    parser.add_argument("--replay-buffer-size", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=".", help="Directory for checkpoints and logs.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint if present.")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--save-buffer-every", type=int, default=10, help="Save replay buffer every N epochs.")
    parser.add_argument("--warmup-games", type=int, default=0, help="Fill buffer with this many warmup games before training.")
    parser.add_argument("--max-gpu-temp", type=float, default=85.0, help="Pause training if GPU temp (C) exceeds this.")
    parser.add_argument("--iterations-per-epoch", type=int, default=10, help="Training iterations per epoch.")
    parser.add_argument("--metrics-db", type=str, default=None, help="Path to training_logs.db (default: checkpoint_dir/training_logs.db).")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP) for faster training.")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"], help="Learning rate scheduler type.")
    args = parser.parse_args()

    if args.auto_config:
        print("Using auto-config from benchmark...")
        args = apply_auto_config(args, args.benchmark_file)
        print(f"   Self-Play Workers: {args.num_workers}, Batch-Size: {args.batch_size}, Replay Buffer: {args.replay_buffer_size}")
    else:
        if args.num_workers is None:
            args.num_workers = 2
        if args.batch_size is None:
            args.batch_size = 32
        if args.replay_buffer_size is None:
            args.replay_buffer_size = 10000

    checkpoint_dir = args.checkpoint_dir
    metrics_db = args.metrics_db or str(Path(checkpoint_dir) / "training_logs.db")
    use_mp = args.num_workers > 1
    metrics_logger = MetricsLogger(db_path=metrics_db, use_mp_queue=use_mp)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will run on CPU and may be very slow.")
        args.use_amp = False  # Disable AMP on CPU

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA: {torch.version.cuda}")
        print(f"AMP (Automatic Mixed Precision): {'Enabled' if args.use_amp else 'Disabled'}")

    config_dict = {k: getattr(args, k) for k in dir(args) if not k.startswith('_') and isinstance(getattr(args, k), (int, float, str, bool))}
    metrics_logger.log_run_meta(device=str(device), config=config_dict)

    if WANDB_AVAILABLE:
        wandb.init(project="archimedes", config=config_dict)

    tpn = TPN().to(device)
    san = SAN().to(device)
    mapper = PlanToMoveMapper().to(device)
    use_ddp = torch.cuda.device_count() > 1 and args.num_workers > 1
    if use_ddp:
        tpn = DDP(tpn)
        san = DDP(san)
        mapper = DDP(mapper)

    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    replay_queue = mp.Queue()

    tpn_params = tpn_params_from(tpn)
    san_params = san_params_from(san)
    mapper_params = mapper_params_from(mapper)
    tpn_optimizer = torch.optim.Adam(tpn_params, lr=0.001)
    san_optimizer = torch.optim.Adam(list(san_params) + list(mapper_params), lr=0.001)
    
    # Initialize AMP GradScaler
    scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
    
    # Learning Rate Schedulers
    total_iters = args.training_iterations
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(tpn_optimizer, T_0=total_iters // 4, T_mult=2)
        scheduler_san = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(san_optimizer, T_0=total_iters // 4, T_mult=2)
        print(f"Using CosineAnnealingWarmRestarts scheduler")
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(tpn_optimizer, mode='min', factor=0.5, patience=10)
        scheduler_san = torch.optim.lr_scheduler.ReduceLROnPlateau(san_optimizer, mode='min', factor=0.5, patience=10)
        print(f"Using ReduceLROnPlateau scheduler")
    else:
        scheduler = None
        scheduler_san = None
        print(f"No scheduler used")

    start_epoch = 0
    training_iteration = 0

    ckpt = load_checkpoint(checkpoint_dir, device) if args.resume else None
    if ckpt is not None:
        tpn_loaded = tpn.module if hasattr(tpn, 'module') else tpn
        san_loaded = san.module if hasattr(san, 'module') else san
        mapper_loaded = mapper.module if hasattr(mapper, 'module') else mapper
        tpn_loaded.load_state_dict(ckpt["tpn_state_dict"])
        san_loaded.load_state_dict(ckpt["san_state_dict"])
        mapper_loaded.load_state_dict(ckpt["mapper_state_dict"])
        if "tpn_optimizer" in ckpt:
            tpn_optimizer.load_state_dict(ckpt["tpn_optimizer"])
        if "san_optimizer" in ckpt:
            san_optimizer.load_state_dict(ckpt["san_optimizer"])
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scheduler_san" in ckpt and scheduler_san is not None:
            scheduler_san.load_state_dict(ckpt["scheduler_san"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        training_iteration = ckpt.get("training_iteration", 0)
        print(f"Resumed from epoch {start_epoch}, iteration {training_iteration}")
        load_replay_buffer_into(replay_buffer, checkpoint_dir)

    # Warmup phase with robust drain
    if args.warmup_games > 0 and len(replay_buffer) < args.batch_size:
        print(f"Warmup: filling buffer with {args.warmup_games} games...")
        warmup_worker(replay_queue, args.warmup_games, device)
        print(f"Draining warmup queue...")
        games_drained = drain_replay_queue(replay_queue, replay_buffer, tpn, san, mapper, device)
        print(f"Warmup done. Buffer size: {len(replay_buffer)}, Games drained: {games_drained}")

    print(f"\nSelf-Play: {args.num_workers} workers, {args.total_games} games...")
    self_play_start = time.time()
    if args.num_workers > 1 and use_ddp:
        mp.spawn(
            self_play_worker,
            args=(args.num_workers, (tpn, san, mapper), replay_queue, args.total_games // args.num_workers, device, use_ddp),
            nprocs=args.num_workers,
            join=True
        )
    else:
        self_play_worker(0, 1, (tpn, san, mapper), replay_queue, args.total_games, device, False)
    self_play_time = time.time() - self_play_start
    print(f"Self-Play finished in {self_play_time:.2f}s")

    training_start = time.time()
    games_processed = 0
    epoch = start_epoch
    pbar = tqdm(initial=training_iteration, total=total_iters, desc="Training")
    epoch_loss_sum = None
    epoch_acc_sum = None
    epoch_count = 0

    while training_iteration < total_iters:
        if args.max_gpu_temp > 0 and check_thermal_throttle(args.max_gpu_temp):
            metrics_logger.log_ticker(f"Thermal throttle: GPU temp >= {args.max_gpu_temp}C, pausing 60s")
            print(f"GPU temp >= {args.max_gpu_temp}C, pausing 60s...")
            time.sleep(60)
            continue

        hw = get_hardware_snapshot()
        metrics_logger.log_hardware(
            timestamp=hw["timestamp"],
            gpu_utilization_pct=hw.get("gpu_utilization_pct"),
            vram_mb=hw.get("vram_mb"),
            gpu_temp_c=hw.get("gpu_temp_c"),
            cpu_load_pct=hw.get("cpu_load_pct"),
            ram_mb=hw.get("ram_mb"),
            disk_io_read=hw.get("disk_io_read"),
            disk_io_write=hw.get("disk_io_write"),
        )

        # Drain replay queue
        games_drained = drain_replay_queue(replay_queue, replay_buffer, tpn, san, mapper, device, max_drain=10)
        games_processed += games_drained

        if len(replay_buffer) >= args.batch_size:
            iter_start = time.time()
            loss_dict = train_step(tpn, san, mapper, replay_buffer, args.batch_size, tpn_optimizer, san_optimizer, device, scaler)
            
            # Update schedulers
            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(loss_dict["tpn_loss"])
                    scheduler_san.step(loss_dict["san_loss"])
                else:
                    scheduler.step()
                    scheduler_san.step()
            
            training_iteration += 1
            lr = tpn_optimizer.param_groups[0]['lr']

            if epoch_loss_sum is None:
                epoch_loss_sum = {k: 0.0 for k in loss_dict}
                epoch_acc_sum = {"top1": 0.0, "top5": 0.0}
                epoch_count = 0
            for k in loss_dict:
                if k in epoch_loss_sum:
                    epoch_loss_sum[k] += loss_dict[k]
            if "top1_accuracy" in loss_dict:
                epoch_acc_sum["top1"] += loss_dict["top1_accuracy"]
                epoch_acc_sum["top5"] += loss_dict["top5_accuracy"]
            epoch_count += 1

            metrics_logger.log_batch(
                epoch=epoch,
                batch_idx=training_iteration,
                total_loss=loss_dict.get("tpn_loss", 0) + loss_dict.get("san_loss", 0),
                policy_loss=loss_dict.get("policy_loss"),
                value_loss=loss_dict.get("value_loss"),
                lr=lr,
                grad_norm=loss_dict.get("grad_norm"),
            )

            pbar.update(1)
            pbar.set_postfix({
                'TPN': f'{loss_dict["tpn_loss"]:.4f}',
                'SAN': f'{loss_dict["san_loss"]:.4f}',
                'LR': f'{lr:.6f}',
                'Buffer': f'{len(replay_buffer)}/{args.replay_buffer_size}'
            })

            iters_this_epoch = training_iteration % args.iterations_per_epoch if args.iterations_per_epoch else 1
            if args.iterations_per_epoch and (training_iteration % args.iterations_per_epoch == 0):
                epoch += 1
                dur = time.time() - training_start
                avg_loss = {k: epoch_loss_sum[k] / epoch_count for k in epoch_loss_sum} if epoch_count else {}
                top1_avg = epoch_acc_sum["top1"] / epoch_count if epoch_count else 0
                top5_avg = epoch_acc_sum["top5"] / epoch_count if epoch_count else 0
                metrics_logger.log_epoch_summary(
                    epoch=epoch,
                    total_loss=avg_loss.get("tpn_loss", 0) + avg_loss.get("san_loss", 0),
                    policy_loss=avg_loss.get("policy_loss"),
                    value_loss=avg_loss.get("value_loss"),
                    a_sfs_loss=avg_loss.get("a_sfs_loss"),
                    plan_policy_loss=avg_loss.get("plan_policy_loss"),
                    train_loss_avg=avg_loss.get("tpn_loss"),
                    val_loss_avg=None,
                    lr=lr,
                    grad_norm=avg_loss.get("grad_norm"),
                    top1_accuracy=top1_avg,
                    top5_accuracy=top5_avg,
                    duration_sec=dur,
                    num_samples=epoch_count * args.batch_size,
                )
                epoch_loss_sum = None
                epoch_acc_sum = None
                epoch_count = 0

                if args.save_every and (epoch % args.save_every == 0):
                    tpn_sd = tpn.module.state_dict() if hasattr(tpn, 'module') else tpn.state_dict()
                    san_sd = san.module.state_dict() if hasattr(san, 'module') else san.state_dict()
                    mapper_sd = mapper.module.state_dict() if hasattr(mapper, 'module') else mapper.state_dict()
                    checkpoint_state = {
                        "tpn_state_dict": tpn_sd,
                        "san_state_dict": san_sd,
                        "mapper_state_dict": mapper_sd,
                        "tpn_optimizer": tpn_optimizer.state_dict(),
                        "san_optimizer": san_optimizer.state_dict(),
                        "epoch": epoch,
                        "training_iteration": training_iteration,
                    }
                    if scheduler is not None:
                        checkpoint_state["scheduler"] = scheduler.state_dict()
                        checkpoint_state["scheduler_san"] = scheduler_san.state_dict()
                    if scaler is not None:
                        checkpoint_state["scaler"] = scaler.state_dict()
                    save_checkpoint_atomic(checkpoint_dir, checkpoint_state)
                    metrics_logger.log_ticker(f"Checkpoint saved at epoch {epoch}")

                if args.save_buffer_every and (epoch % args.save_buffer_every == 0):
                    save_replay_buffer(replay_buffer, checkpoint_dir)
                    metrics_logger.log_ticker(f"Replay buffer saved at epoch {epoch}")

    pbar.close()
    total_time = time.time() - training_start
    print(f"\nTraining done. Time: {total_time:.2f}s, Games processed: {games_processed}, Iterations: {training_iteration}")

    tpn_sd = tpn.module.state_dict() if hasattr(tpn, 'module') else tpn.state_dict()
    san_sd = san.module.state_dict() if hasattr(san, 'module') else san.state_dict()
    mapper_sd = mapper.module.state_dict() if hasattr(mapper, 'module') else mapper.state_dict()
    final_checkpoint = {
        "tpn_state_dict": tpn_sd,
        "san_state_dict": san_sd,
        "mapper_state_dict": mapper_sd,
        "tpn_optimizer": tpn_optimizer.state_dict(),
        "san_optimizer": san_optimizer.state_dict(),
        "epoch": epoch,
        "training_iteration": training_iteration,
    }
    if scheduler is not None:
        final_checkpoint["scheduler"] = scheduler.state_dict()
        final_checkpoint["scheduler_san"] = scheduler_san.state_dict()
    if scaler is not None:
        final_checkpoint["scaler"] = scaler.state_dict()
    save_checkpoint_atomic(checkpoint_dir, final_checkpoint)
    metrics_logger.flush()
    metrics_logger.shutdown()


if __name__ == "__main__":
    main()

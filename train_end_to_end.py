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

from src.archimedes.model import TPN, SAN, PlanToMoveMapper
from src.archimedes.search import ConceptualGraphSearch
from src.archimedes.rewards import calculate_sfs
from src.archimedes.representation import board_to_tensor
from src.archimedes.utils import move_to_index

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def self_play_worker(rank, world_size, model, replay_queue, num_games):
    setup(rank, world_size)
    tpn, san, mapper = model
    search = ConceptualGraphSearch(tpn, san, mapper)

    for _ in range(num_games):
        board = chess.Board()
        game_history = []
        while not board.is_game_over(claim_draw=True):
            search_result = search.search(board)
            game_history.append((board.fen(), search_result["best_move"]))
            board.push(search_result["best_move"])

        result = board.result(claim_draw=True)
        final_game_result = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)

        if rank == 0:
            replay_queue.put((game_history, final_game_result))

    cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_step(tpn, san, mapper, replay_buffer, batch_size, tpn_optimizer, san_optimizer):
    experiences = replay_buffer.sample(batch_size)

    board_tensors = torch.stack([exp["board_tensor"] for exp in experiences])
    final_policies = torch.cat([exp["final_policy"] for exp in experiences])
    v_tacticals = torch.cat([exp["v_tactical"] for exp in experiences])
    a_sfs_predictions = torch.cat([exp["a_sfs_prediction"] for exp in experiences])
    game_results = torch.tensor([exp["final_game_result"] for exp in experiences]).unsqueeze(1)
    best_move_indices = torch.tensor([exp["best_move_index"] for exp in experiences])
    plan_policies = torch.cat([exp["plan_policy"] for exp in experiences])

    real_sfs_list = [calculate_sfs(exp["board_after_plan"], exp["original_goal_vector"], tpn, san) for exp in experiences]
    real_sfs = torch.tensor(real_sfs_list, dtype=torch.float32).unsqueeze(1)

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

def main():
    parser = argparse.ArgumentParser(description="End-to-end training for Archimedes.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of self-play workers.")
    parser.add_argument("--total-games", type=int, default=10, help="Total number of self-play games to generate.")
    args = parser.parse_args()

    wandb.init(project="archimedes", config=args)

    tpn = DDP(TPN())
    san = DDP(SAN())
    mapper = DDP(PlanToMoveMapper())

    replay_buffer = ReplayBuffer(10000)
    replay_queue = mp.Queue()

    tpn_optimizer = torch.optim.Adam(tpn.parameters())
    san_optimizer = torch.optim.Adam(list(san.parameters()) + list(mapper.parameters()))

    world_size = args.num_workers
    mp.spawn(self_play_worker,
             args=(world_size, (tpn, san, mapper), replay_queue, args.total_games // world_size),
             nprocs=world_size,
             join=True)

    while not replay_queue.empty():
        game_history, final_game_result = replay_queue.get()
        board = chess.Board()
        for fen, move in game_history:
            board.set_fen(fen)
            search_result = ConceptualGraphSearch(tpn, san, mapper).search(board)

            experience = {
                "board_tensor": board_to_tensor(board), "final_policy": search_result["final_policy"],
                "v_tactical": search_result["v_tactical"], "a_sfs_prediction": search_result["a_sfs_prediction"],
                "original_goal_vector": search_result["original_goal_vector"],
                "board_after_plan": search_result["board_after_plan"],
                "best_move_index": move_to_index(move),
                "plan_policy": search_result["plan_policy"],
                "final_game_result": final_game_result
            }
            replay_buffer.push(experience)
            final_game_result *= -1

        if len(replay_buffer) > 32:
            train_step(tpn, san, mapper, replay_buffer, 32, tpn_optimizer, san_optimizer)

if __name__ == "__main__":
    main()

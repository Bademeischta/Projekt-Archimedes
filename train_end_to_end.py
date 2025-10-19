import torch
import torch.nn as nn
import chess
import argparse
from collections import deque
import random

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

def main():
    parser = argparse.ArgumentParser(description="End-to-end training for Archimedes.")
    parser.add_argument("--num-games", type=int, default=1, help="Number of self-play games to run.")
    args = parser.parse_args()

    tpn = TPN()
    san = SAN()
    mapper = PlanToMoveMapper()

    search = ConceptualGraphSearch(tpn, san, mapper)
    replay_buffer = ReplayBuffer(10000)

    for game_num in range(args.num_games):
        board = chess.Board()
        game_history = []

        while not board.is_game_over(claim_draw=True):
            search_result = search.search(board)

            # Store experience
            experience = {
                "board_tensor": board_to_tensor(board),
                "final_policy": search_result["final_policy"],
                "v_tactical": search_result["v_tactical"],
                "a_sfs_prediction": search_result["a_sfs_prediction"],
                "original_goal_vector": search_result["original_goal_vector"],
                "board_after_plan": search_result["board_after_plan"],
                "best_move_index": move_to_index(search_result["best_move"]),
                "plan_policy": search_result["plan_policy"],
            }
            game_history.append(experience)

            board.push(search_result["best_move"])

        # Game finished, assign results and push to replay buffer
        result = board.result(claim_draw=True)
        if result == "1-0":
            final_game_result = 1.0
        elif result == "0-1":
            final_game_result = -1.0
        else: # Draw
            final_game_result = 0.0

        for experience in reversed(game_history):
            experience["final_game_result"] = final_game_result
            replay_buffer.push(experience)
            # Alternate the result for the other player's perspective
            final_game_result = -final_game_result

        print(f"Game {game_num + 1} finished. Result: {result}. Buffer size: {len(replay_buffer)}")

        print(f"Game {game_num + 1} finished in {board.fullmove_number} moves. Result: {result}. "
              f"Buffer size: {len(replay_buffer)}. "
              f"Tactical Overrides: {search.tactical_overrides}")
        search.tactical_overrides = 0 # Reset for next game

        # Train if buffer is large enough
        if len(replay_buffer) > 32: # Aribtrary batch size
            train_step(tpn, san, mapper, replay_buffer, 32,
                       torch.optim.Adam(tpn.parameters()),
                       torch.optim.Adam(list(san.parameters()) + list(mapper.parameters())))


def train_step(tpn, san, mapper, replay_buffer, batch_size, tpn_optimizer, san_optimizer):
    experiences = replay_buffer.sample(batch_size)

    # Batch data
    board_tensors = torch.stack([exp["board_tensor"] for exp in experiences])
    final_policies = torch.cat([exp["final_policy"] for exp in experiences])
    v_tacticals = torch.cat([exp["v_tactical"] for exp in experiences])
    a_sfs_predictions = torch.cat([exp["a_sfs_prediction"] for exp in experiences])
    game_results = torch.tensor([exp["final_game_result"] for exp in experiences]).unsqueeze(1)
    best_move_indices = torch.tensor([exp["best_move_index"] for exp in experiences])
    plan_policies = torch.cat([exp["plan_policy"] for exp in experiences])

    # Calculate real SFS for the batch
    real_sfs_list = [calculate_sfs(exp["board_after_plan"], exp["original_goal_vector"], tpn, san) for exp in experiences]
    real_sfs = torch.tensor(real_sfs_list, dtype=torch.float32).unsqueeze(1)

    # TPN Loss
    tpn_optimizer.zero_grad()
    policy_loss = torch.nn.CrossEntropyLoss()(final_policies, best_move_indices)
    value_loss = torch.nn.MSELoss()(v_tacticals, game_results)
    tpn_loss = policy_loss + value_loss
    tpn_loss.backward(retain_graph=True) # Retain graph for SAN backward pass
    tpn_optimizer.step()

    # SAN Loss
    san_optimizer.zero_grad()
    a_sfs_loss = torch.nn.MSELoss()(a_sfs_predictions, real_sfs)
    # Policy gradient for plan policy
    advantages = real_sfs - a_sfs_predictions.detach() # Use advantage to reduce variance
    plan_policy_loss = -torch.mean(torch.log_softmax(plan_policies, dim=1) * advantages)
    san_loss = a_sfs_loss + plan_policy_loss
    san_loss.backward()
    san_optimizer.step()

    print(f"Training Step | TPN Loss: {tpn_loss.item():.4f} | SAN Loss: {san_loss.item():.4f}")


if __name__ == "__main__":
    main()

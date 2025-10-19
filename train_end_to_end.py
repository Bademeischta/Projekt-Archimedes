import torch
import torch.nn as nn
import chess
import argparse

from src.archimedes.model import TPN, SAN, PlanToMoveMapper
from src.archimedes.search import ConceptualGraphSearch
from src.archimedes.rewards import calculate_sfs

def main():
    parser = argparse.ArgumentParser(description="End-to-end training skeleton for Archimedes.")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of training steps to simulate.")
    args = parser.parse_args()

    # Initialize models
    tpn = TPN()
    san = SAN()
    mapper = PlanToMoveMapper()

    # Optimizer for the SAN model (specifically for the A-SFS head)
    optimizer = torch.optim.Adam(san.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Initialize the search
    search = ConceptualGraphSearch(tpn, san, mapper)
    board = chess.Board()

    print("Starting end-to-end training skeleton...")

    for step in range(args.num_steps):
        if board.is_game_over():
            print("Game over. Resetting board.")
            board.reset()

        # 1. Use search to get a move and training context
        search_result = search.search(board)

        # 2. Play the move
        board.push(search_result["best_move"])

        # 3. Calculate the "real" SFS reward
        real_sfs = calculate_sfs(
            board_after_plan=search_result["board_after_plan"],
            original_goal_vector=search_result["original_goal_vector"],
            tpn=tpn,
            san=san
        )
        real_sfs_tensor = torch.tensor([real_sfs], dtype=torch.float32)

        # 4. Get the predicted SFS from the search result
        predicted_sfs = search_result["a_sfs_prediction"]

        # 5. Calculate loss and perform backpropagation
        optimizer.zero_grad()
        loss = loss_fn(predicted_sfs, real_sfs_tensor)
        # In a real training loop, other losses would be added here
        # loss.backward() # This would fail without a real autograd graph
        optimizer.step()

        print(f"Step {step + 1}/{args.num_steps} | "
              f"Move: {search_result['best_move'].uci()} | "
              f"Predicted SFS: {predicted_sfs.item():.4f} | "
              f"Real SFS: {real_sfs:.4f} | "
              f"Loss: {loss.item():.4f}")

    print("\nTraining skeleton finished.")

if __name__ == "__main__":
    main()

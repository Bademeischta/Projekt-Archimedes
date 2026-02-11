import torch
import chess
import torch.nn.functional as F

from .model import TPN, SAN
from .representation import board_to_tensor, board_to_graph

def calculate_sfs(
    board_after_plan: chess.Board,
    original_goal_vector: torch.Tensor,
    tpn: TPN,
    san: SAN,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0
) -> float:
    """
    Calculates the Strategic Fulfillment Score (SFS) for a given plan.
    """
    # Determine device from models
    device = next(tpn.parameters()).device

    # 1. S_goal: Goal Achievement
    graph_after_plan = board_to_graph(board_after_plan).to(device)
    with torch.no_grad():
        new_goal_vector, _, _, _ = san(graph_after_plan)

    s_goal = F.cosine_similarity(original_goal_vector, new_goal_vector, dim=1).item()

    # 2. S_resilience: Resilience (Stresstest)
    # Get the initial tactical value of the position
    tensor_after_plan = board_to_tensor(board_after_plan).unsqueeze(0).to(device)
    with torch.no_grad():
        _, initial_v_tactical = tpn(tensor_after_plan)
    initial_v_tactical = initial_v_tactical.item()

    # Simulate 1-2 counter-moves and find the worst-case tactical value
    # FIXED: Batch-process all counter-moves instead of O(n) individual TPN calls
    worst_v_tactical = initial_v_tactical
    if not board_after_plan.is_game_over():
        # Collect all counter-move positions as a batch
        counter_tensors = []
        legal_moves_list = list(board_after_plan.legal_moves)
        for move in legal_moves_list:
            board_after_plan.push(move)
            counter_tensors.append(board_to_tensor(board_after_plan))
            board_after_plan.pop()
        
        if counter_tensors:
            # Batch inference: single TPN call for all counter-moves
            counter_batch = torch.stack(counter_tensors).to(device)
            with torch.no_grad():
                _, v_tactical_opponents = tpn(counter_batch)
            # Find worst-case: opponent's best response minimizes our value
            worst_v_tactical = -v_tactical_opponents.min().item()
        else:
            worst_v_tactical = initial_v_tactical

    # Resilience is the negative of the drop in value (higher is better)
    s_resilience = -(initial_v_tactical - worst_v_tactical)

    # 3. S_initiative: Initiative
    # A simple proxy: the number of legal moves for the opponent. Fewer is better.
    # We normalize it to a 0-1 range, where 1 is high initiative.
    num_legal_moves = board_after_plan.legal_moves.count()
    s_initiative = 1.0 - (min(num_legal_moves, 20) / 20.0) # Cap at 20 moves

    # 4. Combine scores
    sfs = w1 * s_goal + w2 * s_resilience + w3 * s_initiative
    return sfs

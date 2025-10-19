import torch
import chess

from .model import TPN, SAN, PlanToMoveMapper
from .representation import board_to_tensor, board_to_graph
from .utils import index_to_move

class ConceptualGraphSearch:
    def __init__(self, tpn: TPN, san: SAN, mapper: PlanToMoveMapper):
        self.tpn = tpn
        self.san = san
        self.mapper = mapper

    def search(self, board: chess.Board) -> chess.Move:
        # 1. Get representations
        tensor_board = board_to_tensor(board).unsqueeze(0)
        graph_board = board_to_graph(board)

        # 2. SAN forward pass to get the plan
        with torch.no_grad():
            _, plan_embeddings, plan_policy = self.san(graph_board)

        # Select the most likely plan
        best_plan_idx = torch.argmax(plan_policy, dim=1).item()
        selected_plan_embedding = plan_embeddings[:, best_plan_idx, :]

        # 3. TPN forward pass to get policy logits
        with torch.no_grad():
            policy_logits, _ = self.tpn(tensor_board)

        # 4. Map plan to move bias
        with torch.no_grad():
            policy_bias = self.mapper(selected_plan_embedding, policy_logits)

        # 5. Combine and get final policy
        final_policy_logits = policy_logits + policy_bias
        final_policy = torch.softmax(final_policy_logits, dim=1)

        # 6. Select and return the best move
        best_move_index = torch.argmax(final_policy, dim=1).item()

        # Convert index to move, ensuring it's legal
        # (In a real search, we would iterate through legal moves)
        for move in board.legal_moves:
            if index_to_move(best_move_index, board) == move:
                return move

        # Fallback to the first legal move if the predicted move is illegal
        # This is a temporary measure for the skeleton.
        return next(iter(board.legal_moves))

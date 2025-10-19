import torch
import chess

from .model import TPN, SAN, PlanToMoveMapper
from .representation import board_to_tensor, board_to_graph
from .utils import index_to_move

class ConceptualGraphSearch:
    def __init__(self, tpn: TPN, san: SAN, mapper: PlanToMoveMapper, tactical_override_threshold: float = -0.8):
        self.tpn = tpn
        self.san = san
        self.mapper = mapper
        self.tactical_override_threshold = tactical_override_threshold
        self.tactical_overrides = 0

    def _check_tactical_override(self, board: chess.Board, v_tactical: torch.Tensor) -> chess.Move | None:
        """
        Performs a 1-ply search to check for immediate tactical threats.
        Returns a move if an override is necessary, otherwise None.
        """
        if v_tactical.item() < self.tactical_override_threshold:
            # Our position is already bad, check if there is a critical threat
            for move in board.legal_moves:
                board.push(move)
                tensor_board = board_to_tensor(board).unsqueeze(0)
                with torch.no_grad():
                    _, v_tactical_opponent = self.tpn(tensor_board)

                # If the opponent's position is winning, it's a critical threat for us
                if v_tactical_opponent.item() > -self.tactical_override_threshold:
                    board.pop()
                    # This move is a blunder, so we should avoid it.
                    # For now, we will just return the first legal move that is NOT this move.
                    # A more sophisticated implementation would remove this move from the search space.
                    safe_moves = [m for m in board.legal_moves if m != move]
                    if safe_moves:
                        self.tactical_overrides += 1
                        return safe_moves[0]
                else:
                    board.pop()
        return None

    def search(self, board: chess.Board) -> dict:
        # 1. Get representations
        tensor_board = board_to_tensor(board).unsqueeze(0)
        graph_board = board_to_graph(board)

        # Initial TPN forward pass for tactical override check
        with torch.no_grad():
            policy_logits, v_tactical = self.tpn(tensor_board)

        # 2. Priority Arbiter (PA) - Tactical Override Check
        override_move = self._check_tactical_override(board, v_tactical)
        if override_move:
            best_move = override_move
            # In a real implementation, we would not have a plan, so we'd need to handle this.
            # For now, we proceed to get a dummy plan for the training context.
            with torch.no_grad():
                goal_vector, _, plan_policy, a_sfs_prediction = self.san(graph_board)
            plan_policy = torch.zeros_like(plan_policy) # Dummy policy
        else:
            # 3. SAN forward pass to get the plan
            with torch.no_grad():
                goal_vector, plan_embeddings, plan_policy, a_sfs_prediction = self.san(graph_board)

            best_plan_idx = torch.argmax(plan_policy, dim=1).item()
            selected_plan_embedding = plan_embeddings[:, best_plan_idx, :]

            # 4. Map plan to move bias
            with torch.no_grad():
                policy_bias = self.mapper(selected_plan_embedding, policy_logits)

            # 5. Combine and get final policy
            final_policy_logits = policy_logits + policy_bias
            final_policy = torch.softmax(final_policy_logits, dim=1)

            # 6. Select the best move from the biased policy
            best_move_index = torch.argmax(final_policy, dim=1).item()
            predicted_move = index_to_move(best_move_index, board)

            if predicted_move in board.legal_moves:
                best_move = predicted_move
            else:
                best_move = next(iter(board.legal_moves))

        # 7. Prepare training context
        board_after_plan = board.copy()
        board_after_plan.push(best_move)

        return {
            "best_move": best_move,
            "a_sfs_prediction": a_sfs_prediction,
            "original_goal_vector": goal_vector,
            "board_after_plan": board_after_plan,
            "final_policy": final_policy if not override_move else torch.zeros_like(policy_logits),
            "v_tactical": v_tactical,
            "plan_policy": plan_policy if not override_move else torch.zeros_like(plan_policy),
        }

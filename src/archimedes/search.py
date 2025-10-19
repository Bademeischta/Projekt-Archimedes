import torch
import chess
import math
import numpy as np

from .model import TPN, SAN, PlanToMoveMapper
from .representation import board_to_tensor, board_to_graph
from .utils import index_to_move, move_to_index

class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class ConceptualGraphSearch:
    def __init__(self, tpn: TPN, san: SAN, mapper: PlanToMoveMapper, tactical_override_threshold: float = -0.8, num_simulations: int = 50, cpuct: float = 1.1):
        self.tpn = tpn
        self.san = san
        self.mapper = mapper
        self.tactical_override_threshold = tactical_override_threshold
        self.tactical_overrides = 0
        self.num_simulations = num_simulations
        self.cpuct = cpuct

    def _check_tactical_override(self, board: chess.Board, v_tactical: torch.Tensor) -> chess.Move | None:
        """
        Performs a batched 1-ply search to check for immediate tactical threats.
        Returns a move to avoid if an override is necessary, otherwise None.
        """
        if v_tactical.item() >= self.tactical_override_threshold:
            return None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Create a batch of next board states
        next_board_tensors = []
        for move in legal_moves:
            board.push(move)
            next_board_tensors.append(board_to_tensor(board))
            board.pop()

        batch = torch.stack(next_board_tensors)

        # Get TPN evaluations for all next states in a single batch
        with torch.no_grad():
            _, v_tactical_opponents = self.tpn(batch)

        # Check if any of the opponent's responses are winning for them
        for i, move in enumerate(legal_moves):
            if v_tactical_opponents[i].item() > -self.tactical_override_threshold:
                # A blunder is found. We must avoid this move.
                self.tactical_overrides += 1
                # Return the first safe move found.
                safe_moves = [m for m in legal_moves if m != move]
                return safe_moves[0] if safe_moves else next(iter(legal_moves)) # Failsafe

        return None

    def search(self, board: chess.Board, temperature: float = 1.0) -> dict:
        root = Node(prior=0)

        # Get initial policy and value from network
        tensor_board = board_to_tensor(board).unsqueeze(0)
        graph_board = board_to_graph(board)
        with torch.no_grad():
            policy_logits, v_tactical = self.tpn(tensor_board)
            goal_vector, plan_embeddings, plan_policy, a_sfs_prediction = self.san(graph_board)

        # Priority Arbiter Check
        override_move = self._check_tactical_override(board, v_tactical)
        if override_move:
            # Simplified handling for override
            return {"best_move": override_move, "a_sfs_prediction": a_sfs_prediction, "original_goal_vector": goal_vector,
                    "board_after_plan": board.copy().push(override_move), "final_policy": torch.zeros_like(policy_logits),
                    "v_tactical": v_tactical, "plan_policy": torch.zeros_like(plan_policy)}

        best_plan_idx = torch.argmax(plan_policy, dim=1).item()
        selected_plan_embedding = plan_embeddings[:, best_plan_idx, :]
        with torch.no_grad():
            policy_bias = self.mapper(selected_plan_embedding, policy_logits)
        final_policy = torch.softmax(policy_logits + policy_bias, dim=1).squeeze(0).cpu().numpy()

        # Expand root
        for move in board.legal_moves:
            root.children[move] = Node(prior=final_policy[move_to_index(move)])

        for _ in range(self.num_simulations):
            self.run_simulation(board.copy(), root)

        # Select move
        visit_counts = np.array([node.visit_count for node in root.children.values()])
        moves = list(root.children.keys())
        if temperature == 0:
            action = np.argmax(visit_counts)
        else:
            action = np.random.choice(len(moves), p=(visit_counts / np.sum(visit_counts)))
        best_move = moves[action]

        # Prepare training context
        board_after_plan = board.copy()
        board_after_plan.push(best_move)

        return {"best_move": best_move, "a_sfs_prediction": a_sfs_prediction, "original_goal_vector": goal_vector,
                "board_after_plan": board_after_plan, "final_policy": torch.from_numpy(final_policy).unsqueeze(0),
                "v_tactical": v_tactical, "plan_policy": plan_policy}

    def run_simulation(self, board: chess.Board, node: Node):
        # Selection
        move, child_node = self.select_child(node, board)
        if move is None:
            return -node.value
        board.push(move)

        # Expansion and Simulation
        if child_node.visit_count == 0:
            tensor_board = board_to_tensor(board).unsqueeze(0)
            with torch.no_grad():
                _, value = self.tpn(tensor_board)
            value = value.item()
        else:
            value = self.run_simulation(board, child_node)

        # Backpropagation
        child_node.visit_count += 1
        child_node.value_sum += value
        return -value

    def select_child(self, node: Node, board: chess.Board):
        sqrt_total_visits = math.sqrt(sum(child.visit_count for child in node.children.values()))

        best_score = -np.inf
        best_move = None
        best_child = None

        legal_moves = {move for move in board.legal_moves}

        for move, child in node.children.items():
            if move not in legal_moves: continue

            ucb_score = child.value + self.cpuct * child.prior * (sqrt_total_visits / (1 + child.visit_count))
            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_move, best_child

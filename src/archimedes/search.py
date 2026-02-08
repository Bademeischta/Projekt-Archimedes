"""
MCTS-based ConceptualGraphSearch with optional TT, Dirichlet noise, and MCTS stats.
"""

import time
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
    def __init__(
        self,
        tpn: TPN,
        san: SAN,
        mapper: PlanToMoveMapper,
        tactical_override_threshold: float = -0.8,
        num_simulations: int = 50,
        cpuct: float = 1.1,
        use_transposition_table: bool = False,
        tt_max_size: int = 10000,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.tpn = tpn
        self.san = san
        self.mapper = mapper
        self.tactical_override_threshold = tactical_override_threshold
        self.tactical_overrides = 0
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.use_transposition_table = use_transposition_table
        self.tt_max_size = tt_max_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self._tt = {} if use_transposition_table else None
        self._tt_hits = 0
        self._tt_misses = 0

    def clear_tt(self):
        """Clear transposition table (call between games or when desired)."""
        if self._tt is not None:
            self._tt.clear()
        self._tt_hits = 0
        self._tt_misses = 0

    def _check_tactical_override(self, board: chess.Board, v_tactical: torch.Tensor) -> chess.Move | None:
        if v_tactical.item() >= self.tactical_override_threshold:
            return None
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        next_board_tensors = []
        for move in legal_moves:
            board.push(move)
            next_board_tensors.append(board_to_tensor(board))
            board.pop()
        batch = torch.stack(next_board_tensors)
        with torch.no_grad():
            _, v_tactical_opponents = self.tpn(batch)
        for i, move in enumerate(legal_moves):
            if v_tactical_opponents[i].item() > -self.tactical_override_threshold:
                self.tactical_overrides += 1
                safe_moves = [m for m in legal_moves if m != move]
                return safe_moves[0] if safe_moves else next(iter(legal_moves))
        return None

    def search(
        self,
        board: chess.Board,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
    ) -> dict:
        root = Node(prior=0)
        search_start = time.perf_counter()

        tensor_board = board_to_tensor(board).unsqueeze(0)
        graph_board = board_to_graph(board)
        with torch.no_grad():
            policy_logits, v_tactical = self.tpn(tensor_board)
            goal_vector, plan_embeddings, plan_policy, a_sfs_prediction = self.san(graph_board)

        override_move = self._check_tactical_override(board, v_tactical)
        if override_move:
            return {
                "best_move": override_move,
                "a_sfs_prediction": a_sfs_prediction,
                "original_goal_vector": goal_vector,
                "board_after_plan": board.copy().push(override_move),
                "final_policy": torch.zeros_like(policy_logits),
                "v_tactical": v_tactical,
                "plan_policy": torch.zeros_like(plan_policy),
                "mcts_stats": {},
            }

        best_plan_idx = torch.argmax(plan_policy, dim=1).item()
        selected_plan_embedding = plan_embeddings[:, best_plan_idx, :]
        with torch.no_grad():
            policy_bias = self.mapper(selected_plan_embedding, policy_logits)
        final_policy = torch.softmax(policy_logits + policy_bias, dim=1).squeeze(0).cpu().numpy()

        moves = list(board.legal_moves)
        for move in moves:
            root.children[move] = Node(prior=final_policy[move_to_index(move)])

        if add_dirichlet_noise and len(moves) > 0:
            n = len(moves)
            noise = np.random.dirichlet(np.full(n, self.dirichlet_alpha))
            for i, move in enumerate(moves):
                root.children[move].prior = (
                    (1 - self.dirichlet_epsilon) * root.children[move].prior
                    + self.dirichlet_epsilon * noise[i]
                )

        depths = []
        puct_exploration = []
        puct_exploitation = []
        sim_stats = []

        for _ in range(self.num_simulations):
            self.run_simulation(board.copy(), root, 0, sim_stats)
        for d, ex, ey in sim_stats:
            if d is not None:
                depths.append(d)
            if ex is not None:
                puct_exploration.append(ex)
            if ey is not None:
                puct_exploitation.append(ey)

        search_elapsed = time.perf_counter() - search_start
        total_nodes = sum(c.visit_count for c in root.children.values())
        nps = total_nodes / search_elapsed if search_elapsed > 0 else 0
        avg_depth = float(np.mean(depths)) if depths else 0
        max_depth = int(max(depths)) if depths else 0
        visit_counts = [c.visit_count for c in root.children.values()]
        branching = len(root.children)
        visit_histogram = np.bincount(visit_counts, minlength=min(max(visit_counts) + 1, 256)).tolist()[:32]
        cache_hit_rate = (
            self._tt_hits / (self._tt_hits + self._tt_misses)
            if (self._tt_hits + self._tt_misses) > 0 else None
        )
        mcts_stats = {
            "avg_depth": avg_depth,
            "max_depth": max_depth,
            "nps": nps,
            "branching_factor": float(branching),
            "cache_hit_rate": cache_hit_rate,
            "puct_exploration_avg": float(np.mean(puct_exploration)) if puct_exploration else None,
            "puct_exploitation_avg": float(np.mean(puct_exploitation)) if puct_exploitation else None,
            "visit_histogram_json": None,
        }
        try:
            import json
            mcts_stats["visit_histogram_json"] = json.dumps(visit_histogram)
        except Exception:
            pass

        if temperature == 0:
            action = np.argmax(visit_counts)
        else:
            action = np.random.choice(len(moves), p=(np.array(visit_counts) / max(sum(visit_counts), 1)))
        best_move = moves[action]

        board_after_plan = board.copy()
        board_after_plan.push(best_move)

        return {
            "best_move": best_move,
            "a_sfs_prediction": a_sfs_prediction,
            "original_goal_vector": goal_vector,
            "board_after_plan": board_after_plan,
            "final_policy": torch.from_numpy(final_policy).unsqueeze(0),
            "v_tactical": v_tactical,
            "plan_policy": plan_policy,
            "mcts_stats": mcts_stats,
        }

    def _tt_key(self, board: chess.Board) -> str:
        return board.fen()

    def run_simulation(self, board: chess.Board, node: Node, depth: int, sim_stats: list = None):
        if sim_stats is None:
            sim_stats = []
        move, child_node = self.select_child(node, board)
        exploration_term = None
        exploitation_term = None
        if move is None:
            return -node.value
        sqrt_n = math.sqrt(sum(c.visit_count for c in node.children.values()))
        if child_node.visit_count == 0:
            exploitation_term = child_node.value
            exploration_term = self.cpuct * child_node.prior * (sqrt_n / (1 + child_node.visit_count))
        board.push(move)

        if child_node.visit_count == 0:
            tt_key = self._tt_key(board) if self._tt is not None else None
            if tt_key is not None and tt_key in self._tt:
                value = self._tt[tt_key]
                self._tt_hits += 1
            else:
                tensor_board = board_to_tensor(board).unsqueeze(0)
                with torch.no_grad():
                    _, value = self.tpn(tensor_board)
                value = value.item()
                self._tt_misses += 1
                if self._tt is not None:
                    if len(self._tt) >= self.tt_max_size:
                        self._tt.clear()
                    self._tt[tt_key] = value
            sim_stats.append((depth + 1, exploration_term, exploitation_term))
        else:
            value = self.run_simulation(board, child_node, depth + 1, sim_stats)

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
            if move not in legal_moves:
                continue
            ucb_score = child.value + self.cpuct * child.prior * (sqrt_total_visits / (1 + child.visit_count))
            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_move, best_child

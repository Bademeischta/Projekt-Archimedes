"""
MCTS-based ConceptualGraphSearch with time management, LRU TT, Q-normalization, and Dirichlet noise.
"""

import time
import torch
import chess
import math
import numpy as np
from collections import OrderedDict

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


class LRUTranspositionTable:
    """
    LRU (Least Recently Used) Transposition Table for MCTS.
    Automatically evicts least recently used entries when full.
    """
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str):
        """Get value and mark as recently used."""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: float):
        """Put value and evict LRU if necessary."""
        if key in self.cache:
            # Update existing entry
            self.cache.move_to_end(key)
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        """Clear the cache and reset stats."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ConceptualGraphSearch:
    def __init__(
        self,
        tpn: TPN,
        san: SAN,
        mapper: PlanToMoveMapper,
        tactical_override_threshold: float = -0.8,
        num_simulations: int = 50,
        time_limit: float = None,  # NEW: Time-based search (seconds)
        cpuct: float = 1.1,
        use_transposition_table: bool = True,  # FIXED: Enable by default for performance
        tt_max_size: int = 10000,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_q_normalization: bool = True,  # NEW: Q-value normalization
    ):
        self.tpn = tpn
        self.san = san
        self.mapper = mapper
        self.tactical_override_threshold = tactical_override_threshold
        self.tactical_overrides = 0
        self.num_simulations = num_simulations
        self.time_limit = time_limit  # If set, overrides num_simulations
        self.cpuct = cpuct
        self.use_transposition_table = use_transposition_table
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_q_normalization = use_q_normalization
        
        # Determine device from models
        self.device = next(tpn.parameters()).device

        # LRU Transposition Table
        self._tt = LRUTranspositionTable(tt_max_size) if use_transposition_table else None
        
        # Q-value normalization tracking
        self.q_min = 0.0
        self.q_max = 1.0

    def clear_tt(self):
        """Clear transposition table (call between games or when desired)."""
        if self._tt is not None:
            self._tt.clear()

    def _normalize_q(self, q_value: float) -> float:
        """Normalize Q-value to [0, 1] using dynamic min-max tracking."""
        if not self.use_q_normalization:
            return q_value
        
        # Update min/max
        self.q_min = min(self.q_min, q_value)
        self.q_max = max(self.q_max, q_value)
        
        # Normalize
        if self.q_max - self.q_min < 1e-8:
            return 0.5
        return (q_value - self.q_min) / (self.q_max - self.q_min)

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
        batch = torch.stack(next_board_tensors).to(self.device)
        with torch.no_grad():
            _, v_tactical_opponents = self.tpn(batch)
        
        # FIXED: Find first truly safe move that doesn't give opponent advantage
        for i, move in enumerate(legal_moves):
            # If opponent's value after this move is below threshold, it's safe
            if v_tactical_opponents[i].item() < -self.tactical_override_threshold:
                self.tactical_overrides += 1
                return move
        
        # If no safe move found, return best of bad options (least bad)
        if len(legal_moves) > 0:
            best_idx = torch.argmax(v_tactical_opponents).item()
            self.tactical_overrides += 1
            return legal_moves[best_idx]
        
        return None

    def search(
        self,
        board: chess.Board,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
    ) -> dict:
        root = Node(prior=0)
        search_start = time.perf_counter()

        tensor_board = board_to_tensor(board).unsqueeze(0).to(self.device)
        graph_board = board_to_graph(board).to(self.device)
        with torch.no_grad():
            policy_logits, v_tactical = self.tpn(tensor_board)
            goal_vector, plan_embeddings, plan_policy, a_sfs_prediction = self.san(graph_board)

        override_move = self._check_tactical_override(board, v_tactical)
        if override_move:
            # FIXED: Correctly return board_after_plan as Board object, not OrderedDict
            board_after_plan = board.copy()
            board_after_plan.push(override_move)
            return {
                "best_move": override_move,
                "a_sfs_prediction": a_sfs_prediction,
                "original_goal_vector": goal_vector,
                "board_after_plan": board_after_plan,
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

        # Time-based iterative deepening OR fixed simulations
        if self.time_limit is not None:
            # Iterative deepening with time limit
            depth = 1
            while (time.perf_counter() - search_start) < self.time_limit:
                # Run simulations for current depth
                sims_this_depth = min(10, self.num_simulations)  # Adaptive
                for _ in range(sims_this_depth):
                    if (time.perf_counter() - search_start) >= self.time_limit:
                        break
                    self.run_simulation(board.copy(), root, 0, sim_stats, max_depth=depth)
                depth += 1
        else:
            # Fixed number of simulations
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
        cache_hit_rate = self._tt.hit_rate if self._tt is not None else None
        
        mcts_stats = {
            "avg_depth": avg_depth,
            "max_depth": max_depth,
            "nps": nps,
            "branching_factor": float(branching),
            "cache_hit_rate": cache_hit_rate,
            "puct_exploration_avg": float(np.mean(puct_exploration)) if puct_exploration else None,
            "puct_exploitation_avg": float(np.mean(puct_exploitation)) if puct_exploitation else None,
            "visit_histogram_json": None,
            "q_min": self.q_min if self.use_q_normalization else None,
            "q_max": self.q_max if self.use_q_normalization else None,
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

    def run_simulation(self, board: chess.Board, node: Node, depth: int, sim_stats: list = None, max_depth: int = None):
        if sim_stats is None:
            sim_stats = []
        
        # Check depth limit for iterative deepening
        if max_depth is not None and depth >= max_depth:
            return -node.value
        
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
            if tt_key is not None:
                cached_value = self._tt.get(tt_key)
                if cached_value is not None:
                    value = cached_value
                else:
                    tensor_board = board_to_tensor(board).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        _, value = self.tpn(tensor_board)
                    value = value.item()
                    self._tt.put(tt_key, value)
            else:
                tensor_board = board_to_tensor(board).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, value = self.tpn(tensor_board)
                value = value.item()
            
            # Normalize Q-value
            value = self._normalize_q(value)
            sim_stats.append((depth + 1, exploration_term, exploitation_term))
        else:
            value = self.run_simulation(board, child_node, depth + 1, sim_stats, max_depth)

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
            # Use normalized Q-value
            q_value = self._normalize_q(child.value) if self.use_q_normalization else child.value
            ucb_score = q_value + self.cpuct * child.prior * (sqrt_total_visits / (1 + child.visit_count))
            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_move, best_child

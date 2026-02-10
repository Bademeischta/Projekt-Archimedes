import chess
import torch
from torch_geometric.data import Data

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess.Board object to a tensor representation.
    Shape: (C, 8, 8)
    OPTIMIZED: Vectorized implementation reduces O(768) → O(64) operations
    """
    # Initialize tensor: 22 channels, 8x8 board
    tensor = torch.zeros(22, 8, 8, dtype=torch.float32)
    
    # Piece mapping for fast lookup: (piece_type, color) → channel
    piece_map = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }
    
    # Single pass: set all pieces (O(64) instead of O(768))
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        channel = piece_map.get((piece.piece_type, piece.color))
        if channel is not None:
            tensor[channel, rank, file] = 1.0

    # Castling rights (4 channels) - fill entire planes
    tensor[12] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[13] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[14] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[15] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En-passant (1 channel)
    if board.ep_square:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[16, rank, file] = 1.0

    # Side to move (1 channel)
    tensor[17] = 1.0 if board.turn == chess.WHITE else 0.0

    # Attack maps (2 channels) - vectorized
    for square in chess.SQUARES:
        rank, file = chess.square_rank(square), chess.square_file(square)
        tensor[18, rank, file] = len(board.attackers(chess.WHITE, square))
        tensor[19, rank, file] = len(board.attackers(chess.BLACK, square))

    # Pin maps (2 channels) - vectorized
    for square in chess.SQUARES:
        rank, file = chess.square_rank(square), chess.square_file(square)
        if board.is_pinned(chess.WHITE, square):
            tensor[20, rank, file] = 1.0
        if board.is_pinned(chess.BLACK, square):
            tensor[21, rank, file] = 1.0

    return tensor


def board_to_graph(board: chess.Board) -> Data:
    """
    Converts a chess.Board object to a graph representation.
    Nodes: 64, one for each square.
    Node Features: Piece type, static square properties.
    Edges: Attacks, defends, pawn_chain, pins (dynamic).
    """
    # 1. Node Features
    node_features = []
    # Center squares for static feature
    center_squares = {chess.D4, chess.E4, chess.D5, chess.E5}
    # King safety zones for static feature
    white_king_safety = {chess.F1, chess.G1, chess.H1, chess.F2, chess.G2, chess.H2}
    black_king_safety = {chess.F8, chess.G8, chess.H8, chess.F7, chess.G7, chess.H7}

    piece_map = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }

    for square_idx in chess.SQUARES:
        features = [0.0] * 16 # 13 for piece type, 3 for static properties

        piece = board.piece_at(square_idx)
        if piece:
            features[piece_map[(piece.piece_type, piece.color)]] = 1.0
        else:
            features[12] = 1.0 # "Empty" feature

        # Static features
        if square_idx in center_squares:
            features[13] = 1.0 # is_center
        if square_idx in white_king_safety:
            features[14] = 1.0 # is_king_safety_zone_white
        if square_idx in black_king_safety:
            features[15] = 1.0 # is_king_safety_zone_black

        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float32)

    # 2. Edge Generation
    edge_lists = {
        "attacks": [], "defends": [], "pawn_chain": [], "pins": []
    }

    # First pass: Find pins, which have priority
    pinned_pieces = {} # map from pinned square to pinner square
    for color in [chess.WHITE, chess.BLACK]:
        king_square = board.king(color)
        if king_square is None: continue
        for square in chess.SQUARES:
            if board.is_pinned(color, square):
                for pinner_square in board.attackers(not color, square):
                    pinner_piece = board.piece_at(pinner_square)
                    if pinner_piece and pinner_piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
                        # Check alignment
                        if chess.square_distance(pinner_square, king_square) == \
                           chess.square_distance(pinner_square, square) + chess.square_distance(square, king_square):
                            edge_lists["pins"].append([pinner_square, square])
                            pinned_pieces[square] = pinner_square
                            break # Found the unique pinner

    # Second pass: Generate other edges
    for u in chess.SQUARES:
        piece_u = board.piece_at(u)
        if not piece_u:
            continue

        # Pawn chain edges
        if piece_u.piece_type == chess.PAWN:
            for v in board.attackers(piece_u.color, u):
                piece_v = board.piece_at(v)
                if piece_v and piece_v.piece_type == chess.PAWN and piece_v.color == piece_u.color:
                    edge_lists["pawn_chain"].append([u, v])

        # Attacks and defends edges
        for v in board.attacks(u):
            # IMPORTANT LOGIC-RULE: If u is pinned by p, do not create an attack edge from u to p.
            if u in pinned_pieces and v == pinned_pieces[u]:
                continue

            piece_v = board.piece_at(v)
            if piece_v and piece_v.color == piece_u.color:
                edge_lists["defends"].append([u, v])
            else: # attacks empty square or opponent piece
                edge_lists["attacks"].append([u, v])

    # Combine edges into a single tensor and create edge types
    edge_index = []
    edge_type = []
    edge_type_map = {"attacks": 0, "defends": 1, "pawn_chain": 2, "pins": 3}

    for type_name, edges in edge_lists.items():
        if edges:
            edge_index.extend(edges)
            edge_type.extend([edge_type_map[type_name]] * len(edges))

    if not edge_index: # Handle boards with no edges
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_type_tensor = torch.empty((0), dtype=torch.long)
    else:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

    return Data(x=x, edge_index=edge_index_tensor, edge_type=edge_type_tensor)

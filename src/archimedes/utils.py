import chess

# This implementation is based on the AlphaZero paper's 8x8x73 move representation.
# Total planes = 56 (queen) + 8 (knight) + 9 (underpromotions) = 73
# Total moves = 64 squares * 73 move types = 4672 indices.

# Helper to map direction to an index for queen moves
_DIRECTIONS = {
    "N": (1, 0), "NE": (1, 1), "E": (0, 1), "SE": (-1, 1),
    "S": (-1, 0), "SW": (-1, -1), "W": (0, -1), "NW": (1, -1)
}
_DIRECTION_PLANE_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Helper for knight moves
_KNIGHT_MOVES = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]
_KNIGHT_MOVE_MAP = {move: i for i, move in enumerate(_KNIGHT_MOVES)}

# Helper for promotion directions
_PROMO_DIRECTIONS = [
    (-1, 1), (0, 1), (1, 1) # Left capture, forward, right capture (from white's perspective)
]


def move_to_index(move: chess.Move) -> int:
    """
    Converts a chess.Move object to an index in the 4672-dimensional action space.
    """
    from_square = move.from_square
    to_square = move.to_square

    from_rank, from_file = chess.square_rank(from_square), chess.square_file(from_square)
    to_rank, to_file = chess.square_rank(to_square), chess.square_file(to_square)

    delta_rank = to_rank - from_rank
    delta_file = to_file - from_file

    # 1. Check for Pawn underpromotions (N, B, R)
    if move.promotion and move.promotion != chess.QUEEN:
        promo_piece_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[move.promotion]

        # Normalize direction for black
        if delta_rank < 0:
            delta_file = -delta_file

        direction_offset = _PROMO_DIRECTIONS.index((delta_file, 1))

        move_type_plane = 64 + direction_offset * 3 + promo_piece_offset
        return from_square * 73 + move_type_plane

    # 2. Check for Knight moves
    if (delta_rank, delta_file) in _KNIGHT_MOVE_MAP:
        move_type_plane = 56 + _KNIGHT_MOVE_MAP[(delta_rank, delta_file)]
        return from_square * 73 + move_type_plane

    # 3. Handle Queen-like moves (including queen promotions)
    distance = max(abs(delta_rank), abs(delta_file))
    unit_delta_rank = delta_rank // distance if distance != 0 else 0
    unit_delta_file = delta_file // distance if distance != 0 else 0

    direction_name = ""
    for name, (dr, df) in _DIRECTIONS.items():
        if dr == unit_delta_rank and df == unit_delta_file:
            direction_name = name
            break

    direction_idx = _DIRECTION_PLANE_ORDER.index(direction_name)
    distance_idx = distance - 1

    move_type_plane = direction_idx * 7 + distance_idx
    return from_square * 73 + move_type_plane

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Converts an index in the 4672-dimensional action space back to a chess.Move object.
    """
    from_square = index // 73
    move_type_plane = index % 73

    from_rank, from_file = chess.square_rank(from_square), chess.square_file(from_square)

    # 1. Underpromotions
    if 64 <= move_type_plane < 73:
        plane_offset = move_type_plane - 64
        direction_idx = plane_offset // 3
        promo_piece_idx = plane_offset % 3

        promo_piece = [chess.KNIGHT, chess.BISHOP, chess.ROOK][promo_piece_idx]
        df, dr = _PROMO_DIRECTIONS[direction_idx]

        # Adjust for black's perspective
        if from_rank == 6: # White pawn on 7th rank
            pass
        elif from_rank == 1: # Black pawn on 2nd rank
            df = -df
            dr = -dr

        to_file = from_file + df
        to_rank = from_rank + dr

        return chess.Move(from_square, chess.square(to_file, to_rank), promotion=promo_piece)

    # 2. Knight moves
    if 56 <= move_type_plane < 64:
        dr, df = _KNIGHT_MOVES[move_type_plane - 56]
        to_rank, to_file = from_rank + dr, from_file + df
        return chess.Move(from_square, chess.square(to_file, to_rank))

    # 3. Queen-like moves
    direction_idx = move_type_plane // 7
    distance_idx = move_type_plane % 7
    distance = distance_idx + 1

    direction_name = _DIRECTION_PLANE_ORDER[direction_idx]
    dr, df = _DIRECTIONS[direction_name]

    to_rank = from_rank + dr * distance
    to_file = from_file + df * distance

    to_square = chess.square(to_file, to_rank)

    # Check for queen promotion
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN and (to_rank == 0 or to_rank == 7):
        return chess.Move(from_square, to_square, promotion=chess.QUEEN)

    return chess.Move(from_square, to_square)

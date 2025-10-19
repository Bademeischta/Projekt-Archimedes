import chess
import chess.engine
import argparse

from src.archimedes.search import ConceptualGraphSearch
from src.archimedes.model import TPN, SAN, PlanToMoveMapper

def play_game(model1_search, model2_search, egtb_path=None):
    board = chess.Board()
    if egtb_path:
        board.egtb_path = egtb_path

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            move = model1_search.search(board)["best_move"]
        else:
            move = model2_search.search(board)["best_move"]
        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Elo of a model.")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to the first model's weights.")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to the second model's weights.")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play.")
    parser.add_argument("--egtb-path", type=str, default=None, help="Path to endgame tablebases.")
    args = parser.parse_args()

    # Load models
    tpn1, san1, mapper1 = TPN(), SAN(), PlanToMoveMapper()
    # In a real scenario, you would load weights from args.model1_path
    search1 = ConceptualGraphSearch(tpn1, san1, mapper1)

    tpn2, san2, mapper2 = TPN(), SAN(), PlanToMoveMapper()
    # In a real scenario, you would load weights from args.model2_path
    search2 = ConceptualGraphSearch(tpn2, san2, mapper2)

    score = 0
    for i in range(args.num_games):
        if i % 2 == 0:
            score += play_game(search1, search2, args.egtb_path)
        else:
            # Switch colors
            score -= play_game(search2, search1, args.egtb_path)
        print(f"Game {i+1}/{args.num_games} finished. Current score: {score}")

    elo_diff = 400 * score / args.num_games
    print(f"\nElo difference: {elo_diff:.2f}")

if __name__ == "__main__":
    main()

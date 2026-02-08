"""
Archimedes Dashboard: Streamlit UI with tabs for training, MCTS, chess, hardware,
Play vs AI, GNN viz, live ticker, downloads, and FEN tool.
Uses @st.cache_resource for model/DB. Optional pyngrok for Colab/public access.
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import sqlite3

# Optional plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Colab detection
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Ngrok: only when IN_COLAB or ARCHIMEDES_PUBLIC_DASHBOARD
USE_NGROK = IN_COLAB or os.environ.get("ARCHIMEDES_PUBLIC_DASHBOARD", "").lower() in ("1", "true", "yes")

DEFAULT_CHECKPOINT_DIR = "."
DEFAULT_DB_PATH = "training_logs.db"


def get_db_path():
    return os.environ.get("ARCHIMEDES_DB_PATH", Path(DEFAULT_CHECKPOINT_DIR) / DEFAULT_DB_PATH)


def get_checkpoint_dir():
    return os.environ.get("ARCHIMEDES_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)


def get_db_connection():
    """Non-cached so that when training creates the DB we can read it after refresh."""
    db_path = str(get_db_path())
    if not Path(db_path).exists():
        return None
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except Exception:
        return None


@st.cache_resource
def load_model(checkpoint_path: str):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    import torch
    from src.archimedes.model import TPN, SAN, PlanToMoveMapper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tpn = TPN()
    san = SAN()
    mapper = PlanToMoveMapper()
    tpn.load_state_dict(ckpt["tpn_state_dict"])
    san.load_state_dict(ckpt["san_state_dict"])
    mapper.load_state_dict(ckpt["mapper_state_dict"])
    tpn.to(device).eval()
    san.to(device).eval()
    mapper.to(device).eval()
    return {"tpn": tpn, "san": san, "mapper": mapper, "device": device}


def run_inference_fen(fen: str, model_dict):
    if not model_dict or not fen.strip():
        return None, None
    import torch
    import chess
    from src.archimedes.representation import board_to_tensor, board_to_graph
    try:
        board = chess.Board(fen)
    except Exception:
        return None, None
    device = model_dict["device"]
    tpn = model_dict["tpn"]
    san = model_dict["san"]
    tensor_board = board_to_tensor(board).unsqueeze(0).to(device)
    graph_board = board_to_graph(board)
    with torch.no_grad():
        policy, value = tpn(tensor_board)
        goal, plan_emb, plan_policy, a_sfs = san(graph_board)
    return value.item(), policy

def run_ai_move(board, model_dict):
    if not model_dict:
        return None
    import chess
    from src.archimedes.search import ConceptualGraphSearch
    from src.archimedes.representation import board_to_tensor
    tpn = model_dict["tpn"]
    san = model_dict["san"]
    mapper = model_dict["mapper"]
    search = ConceptualGraphSearch(tpn, san, mapper)
    result = search.search(board, temperature=0, add_dirichlet_noise=False)
    return result["best_move"]


# Unicode pieces for simple board
PIECES = {
    "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔", "P": "♙",
    "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚", "p": "♟",
}


def render_simple_board(fen: str):
    import chess
    try:
        board = chess.Board(fen)
    except Exception:
        return
    lines = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            sq = chess.square(file, rank)
            p = board.piece_at(sq)
            if p:
                sym = PIECES.get(p.symbol(), p.symbol())
                row.append(sym)
            else:
                row.append("·")
        lines.append(" ".join(row))
    return "\n".join(lines)


def tab_training_nn(conn):
    if conn is None:
        st.info("No database found. Start training to generate logs.")
        return
    try:
        df = pd.read_sql_query("SELECT * FROM epoch_summary ORDER BY epoch", conn)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        st.info("No epoch summary data yet.")
        return
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for col in ["total_loss", "policy_loss", "value_loss"]:
            if col in df.columns and df[col].notna().any():
                fig.add_trace(go.Scatter(x=df["epoch"], y=df[col], mode="lines", name=col))
        fig.update_layout(title="Loss over epochs", xaxis_title="Epoch")
        st.plotly_chart(fig, use_container_width=True)
        if "top1_accuracy" in df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["epoch"], y=df["top1_accuracy"], mode="lines", name="Top-1"))
            if "top5_accuracy" in df.columns:
                fig2.add_trace(go.Scatter(x=df["epoch"], y=df["top5_accuracy"], mode="lines", name="Top-5"))
            st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df.tail(50), use_container_width=True)


def tab_mcts(conn):
    if conn is None:
        st.info("No database found.")
        return
    try:
        df = pd.read_sql_query("SELECT * FROM mcts_stats ORDER BY id DESC LIMIT 500", conn)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        st.info("No MCTS stats yet.")
        return
    if PLOTLY_AVAILABLE and "nps" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["nps"].dropna(), mode="lines", name="NPS"))
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.head(100), use_container_width=True)


def tab_chess(conn):
    if conn is None:
        st.info("No database found.")
        return
    try:
        df = pd.read_sql_query("SELECT * FROM game_results ORDER BY id DESC LIMIT 200", conn)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        st.info("No game results yet.")
        return
    st.dataframe(df, use_container_width=True)


def tab_hardware(conn):
    if conn is None:
        st.info("No database found.")
        return
    try:
        df = pd.read_sql_query("SELECT * FROM hardware_snapshots ORDER BY id DESC LIMIT 200", conn)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        st.info("No hardware snapshots yet.")
        return
    if PLOTLY_AVAILABLE and "gpu_temp_c" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["gpu_temp_c"], mode="lines", name="GPU temp °C"))
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.head(50), use_container_width=True)


def tab_play_vs_ai(model_dict):
    import chess
    if "board_fen" not in st.session_state:
        st.session_state["board_fen"] = chess.Board().fen()
    if "game_over" not in st.session_state:
        st.session_state["game_over"] = False

    st.subheader("Play vs AI")
    checkpoint_dir = get_checkpoint_dir()
    ckpt_path = str(Path(checkpoint_dir) / "latest_checkpoint.pt")
    if not Path(ckpt_path).exists():
        st.warning("No checkpoint found. Train a model first.")
        return

    if model_dict is None:
        model_dict = load_model(ckpt_path)
    if model_dict is None:
        st.error("Failed to load model.")
        return

    fen = st.text_input("FEN (or leave default)", value=st.session_state["board_fen"])
    try:
        board = chess.Board(fen)
    except Exception:
        st.error("Invalid FEN")
        return

    st.text("Board (current position):")
    st.code(render_simple_board(board.fen()), language=None)

    if board.is_game_over(claim_draw=True):
        st.session_state["game_over"] = True
        st.success(f"Game over: {board.result(claim_draw=True)}")
        if st.button("New game"):
            st.session_state["board_fen"] = chess.Board().fen()
            st.session_state["game_over"] = False
            st.rerun()
        return

    col1, col2 = st.columns(2)
    with col1:
        move_uci = st.text_input("Your move (e.g. e2e4)", key="move_uci")
        if st.button("Play move"):
            if move_uci:
                try:
                    m = chess.Move.from_uci(move_uci.strip())
                    if m in board.legal_moves:
                        board.push(m)
                        st.session_state["board_fen"] = board.fen()
                        st.rerun()
                    else:
                        st.error("Illegal move")
                except Exception as e:
                    st.error(str(e))
    with col2:
        if st.button("Get AI move"):
            with st.spinner("AI thinking..."):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(run_ai_move, board.copy(), model_dict)
                    best_move = future.result(timeout=60)
                if best_move is not None:
                    board.push(best_move)
                    st.session_state["board_fen"] = board.fen()
                    st.success(f"AI plays: {best_move.uci()}")
                    st.rerun()
                else:
                    st.error("AI could not produce a move.")


def tab_gnn_viz(model_dict):
    st.subheader("GNN / Board visualization")
    fen = st.text_input("FEN for visualization", value="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    if not fen.strip():
        return
    st.code(render_simple_board(fen), language=None)
    try:
        from src.archimedes.representation import board_to_graph
        import chess
        board = chess.Board(fen)
        g = board_to_graph(board)
        st.write("Graph: nodes =", g.x.shape[0], ", edges =", g.edge_index.shape[1])
        if g.x.numel() <= 64 * 20:
            st.write("Node features (sample):", g.x[:8].tolist())
    except Exception as e:
        st.warning(str(e))


def tab_ticker(conn):
    if conn is None:
        st.info("No database found.")
        return
    try:
        df = pd.read_sql_query("SELECT * FROM ticker_events ORDER BY id DESC LIMIT 100", conn)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        st.info("No ticker events yet.")
        return
    for _, row in df.iterrows():
        ts = row.get("timestamp", "")
        msg = row.get("message", "")
        st.text(f"[{ts}] {msg}")


def tab_downloads():
    st.subheader("Downloads")
    checkpoint_dir = get_checkpoint_dir()
    ckpt_path = Path(checkpoint_dir) / "latest_checkpoint.pt"
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            st.download_button("Download latest_checkpoint.pt", f, file_name="latest_checkpoint.pt")
    else:
        st.info("No checkpoint file found.")
    pgn_dir = Path(checkpoint_dir) / "pgn"
    if pgn_dir.exists():
        pgn_files = list(pgn_dir.glob("*.pgn"))
        for p in pgn_files[:10]:
            with open(p, "r") as f:
                st.download_button(f"Download {p.name}", f.read(), file_name=p.name, key=str(p))


def tab_tools(model_dict):
    st.subheader("FEN → Value")
    fen = st.text_area("FEN position", value="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    if st.button("Compute value"):
        checkpoint_dir = get_checkpoint_dir()
        ckpt_path = str(Path(checkpoint_dir) / "latest_checkpoint.pt")
        model_dict = load_model(ckpt_path) if Path(ckpt_path).exists() else None
        if model_dict is None:
            st.warning("Load a checkpoint first.")
            return
        import torch
        value, policy = run_inference_fen(fen, model_dict)
        if value is not None:
            st.metric("Value (TPN)", f"{value:.4f}")
            st.caption("Value in [-1, 1]: positive = better for White.")
        else:
            st.error("Invalid FEN or inference failed.")


def main():
    st.set_page_config(page_title="Archimedes Dashboard", layout="wide")
    st.title("Archimedes Dashboard")

    conn = get_db_connection()
    checkpoint_dir = get_checkpoint_dir()
    ckpt_path = str(Path(checkpoint_dir) / "latest_checkpoint.pt")
    model_dict = load_model(ckpt_path) if Path(ckpt_path).exists() else None

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Training & NN", "MCTS & Suche", "Schach", "Hardware",
        "Play vs AI", "GNN-Visualisierung", "Live-Ticker", "Downloads", "Tools"
    ])
    with tab1:
        tab_training_nn(conn)
    with tab2:
        tab_mcts(conn)
    with tab3:
        tab_chess(conn)
    with tab4:
        tab_hardware(conn)
    with tab5:
        tab_play_vs_ai(model_dict)
    with tab6:
        tab_gnn_viz(model_dict)
    with tab7:
        tab_ticker(conn)
    with tab8:
        tab_downloads()
    with tab9:
        tab_tools(model_dict)


if __name__ == "__main__":
    main()

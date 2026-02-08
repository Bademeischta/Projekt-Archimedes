"""
Asynchronous MetricsLogger for Archimedes training.
Uses a queue + background writer; supports multiprocessing.Queue when workers log.
SQLite with WAL mode for concurrent read (dashboard) / write (trainer).
No blobs in DB - large tensors go to logs/tensors/ as files; only paths or scalars in DB.
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from queue import Empty, Queue

# Optional: use multiprocessing.Queue when multi-process self-play workers log
try:
    import multiprocessing
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


def _writer_loop(db_path: str, log_queue, stop_event, flush_event, use_mp_queue: bool):
    """Background loop: read from queue, write to SQLite. WAL mode enabled."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")

    while not stop_event.is_set():
        try:
            try:
                msg = log_queue.get(timeout=0.5)
            except (Empty, Exception):
                msg = None
            if msg is not None:
                _apply_message(conn, msg)
            if flush_event.is_set():
                flush_event.clear()
        except Exception as e:
            import sys
            print(f"[MetricsLogger] Writer error: {e}", file=sys.stderr)

    conn.commit()
    conn.close()


def _apply_message(conn: sqlite3.Connection, msg: dict):
    kind = msg.get("_kind")
    if kind == "run_meta":
        conn.execute(
            """INSERT INTO run_meta (run_id, start_time, device, config_json)
               VALUES (?, ?, ?, ?)""",
            (msg["run_id"], msg["start_time"], msg.get("device", ""), msg.get("config_json", "{}"))
        )
    elif kind == "epoch_summary":
        conn.execute(
            """INSERT INTO epoch_summary (
                run_id, epoch, total_loss, policy_loss, value_loss, a_sfs_loss, plan_policy_loss,
                train_loss_avg, val_loss_avg, lr, grad_norm, top1_accuracy, top5_accuracy,
                duration_sec, num_samples
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg["run_id"], msg["epoch"], msg.get("total_loss"), msg.get("policy_loss"),
                msg.get("value_loss"), msg.get("a_sfs_loss"), msg.get("plan_policy_loss"),
                msg.get("train_loss_avg"), msg.get("val_loss_avg"), msg.get("lr"),
                msg.get("grad_norm"), msg.get("top1_accuracy"), msg.get("top5_accuracy"),
                msg.get("duration_sec"), msg.get("num_samples")
            )
        )
    elif kind == "batch_log":
        conn.execute(
            """INSERT INTO batch_log (run_id, epoch, batch_idx, total_loss, policy_loss, value_loss, lr, grad_norm)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg["run_id"], msg["epoch"], msg["batch_idx"], msg.get("total_loss"), msg.get("policy_loss"),
             msg.get("value_loss"), msg.get("lr"), msg.get("grad_norm"))
        )
    elif kind == "gnn_internals":
        conn.execute(
            """INSERT INTO gnn_internals (run_id, epoch, batch_idx, layer_name, weight_mean, weight_std, weight_min, weight_max, tensor_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg["run_id"], msg["epoch"], msg.get("batch_idx"), msg.get("layer_name"),
             msg.get("weight_mean"), msg.get("weight_std"), msg.get("weight_min"), msg.get("weight_max"),
             msg.get("tensor_path"))
        )
    elif kind == "mcts_stats":
        conn.execute(
            """INSERT INTO mcts_stats (
                run_id, epoch, game_idx, move_idx, avg_depth, max_depth, nps,
                branching_factor, cutoff_rate, cache_hit_rate, puct_exploration_avg, puct_exploitation_avg,
                visit_histogram_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg["run_id"], msg.get("epoch"), msg.get("game_idx"), msg.get("move_idx"),
             msg.get("avg_depth"), msg.get("max_depth"), msg.get("nps"), msg.get("branching_factor"),
             msg.get("cutoff_rate"), msg.get("cache_hit_rate"), msg.get("puct_exploration_avg"),
             msg.get("puct_exploitation_avg"), msg.get("visit_histogram_json"))
        )
    elif kind == "game_results":
        conn.execute(
            """INSERT INTO game_results (
                run_id, epoch, game_id, result, termination, color_white_elo_estimate, color_black_elo_estimate,
                avg_game_length, opening_eco, white_wins, black_wins, draws
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg["run_id"], msg.get("epoch"), msg.get("game_id"), msg.get("result"), msg.get("termination"),
             msg.get("color_white_elo_estimate"), msg.get("color_black_elo_estimate"), msg.get("avg_game_length"),
             msg.get("opening_eco"), msg.get("white_wins"), msg.get("black_wins"), msg.get("draws"))
        )
    elif kind == "blunder_analysis":
        conn.execute(
            """INSERT INTO blunder_analysis (run_id, epoch, game_id, blunder_rate, centipawn_loss_vs_stockfish, centipawn_loss_vs_teacher)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (msg["run_id"], msg.get("epoch"), msg.get("game_id"), msg.get("blunder_rate"),
             msg.get("centipawn_loss_vs_stockfish"), msg.get("centipawn_loss_vs_teacher"))
        )
    elif kind == "endgame_stats":
        conn.execute(
            """INSERT INTO endgame_stats (run_id, epoch, mate_in_x_errors_json)
               VALUES (?, ?, ?)""",
            (msg["run_id"], msg.get("epoch"), msg.get("mate_in_x_errors_json"))
        )
    elif kind == "hardware_snapshots":
        conn.execute(
            """INSERT INTO hardware_snapshots (
                run_id, timestamp, gpu_utilization_pct, vram_mb, gpu_temp_c, cpu_load_pct, ram_mb,
                disk_io_read, disk_io_write, positions_per_watt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg["run_id"], msg.get("timestamp"), msg.get("gpu_utilization_pct"), msg.get("vram_mb"),
             msg.get("gpu_temp_c"), msg.get("cpu_load_pct"), msg.get("ram_mb"), msg.get("disk_io_read"),
             msg.get("disk_io_write"), msg.get("positions_per_watt"))
        )
    elif kind == "ticker_events":
        conn.execute(
            """INSERT INTO ticker_events (run_id, timestamp, message) VALUES (?, ?, ?)""",
            (msg["run_id"], msg.get("timestamp"), msg.get("message", ""))
        )
    conn.commit()


def _create_schema(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_meta (
            run_id TEXT PRIMARY KEY,
            start_time REAL,
            device TEXT,
            config_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS epoch_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            epoch INTEGER,
            total_loss REAL, policy_loss REAL, value_loss REAL, a_sfs_loss REAL, plan_policy_loss REAL,
            train_loss_avg REAL, val_loss_avg REAL, lr REAL, grad_norm REAL,
            top1_accuracy REAL, top5_accuracy REAL, duration_sec REAL, num_samples INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, epoch INTEGER, batch_idx INTEGER,
            total_loss REAL, policy_loss REAL, value_loss REAL, lr REAL, grad_norm REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gnn_internals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, epoch INTEGER, batch_idx INTEGER, layer_name TEXT,
            weight_mean REAL, weight_std REAL, weight_min REAL, weight_max REAL,
            tensor_path TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mcts_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, epoch INTEGER, game_idx INTEGER, move_idx INTEGER,
            avg_depth REAL, max_depth INTEGER, nps REAL, branching_factor REAL, cutoff_rate REAL,
            cache_hit_rate REAL, puct_exploration_avg REAL, puct_exploitation_avg REAL,
            visit_histogram_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, epoch INTEGER, game_id INTEGER, result TEXT, termination TEXT,
            color_white_elo_estimate REAL, color_black_elo_estimate REAL, avg_game_length REAL,
            opening_eco TEXT, white_wins INTEGER, black_wins INTEGER, draws INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blunder_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, epoch INTEGER, game_id INTEGER, blunder_rate REAL,
            centipawn_loss_vs_stockfish REAL, centipawn_loss_vs_teacher REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS endgame_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, epoch INTEGER, mate_in_x_errors_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, timestamp REAL, gpu_utilization_pct REAL, vram_mb REAL, gpu_temp_c REAL,
            cpu_load_pct REAL, ram_mb REAL, disk_io_read REAL, disk_io_write REAL, positions_per_watt REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, timestamp REAL, message TEXT
        )
    """)
    conn.commit()


class MetricsLogger:
    """
    Asynchronous metrics logger. Put log dicts on the queue; a background thread
    writes to SQLite. Use multiprocessing.Queue when self-play workers (other processes) log.
    """

    def __init__(self, db_path: str = "training_logs.db", use_mp_queue: bool = False, run_id: str = None):
        self.db_path = db_path
        self.use_mp_queue = use_mp_queue and MP_AVAILABLE
        self.run_id = run_id or f"run_{int(time.time())}_{os.getpid()}"

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        _create_schema(conn)
        conn.close()

        if self.use_mp_queue:
            self._queue = multiprocessing.Queue()
        else:
            self._queue = Queue()

        self._stop = threading.Event()
        self._flush_event = threading.Event()
        self._writer = threading.Thread(
            target=_writer_loop,
            args=(db_path, self._queue, self._stop, self._flush_event, self.use_mp_queue),
            daemon=True,
        )
        self._writer.start()

    def _put(self, msg: dict):
        msg["run_id"] = msg.get("run_id") or self.run_id
        if self.use_mp_queue:
            self._queue.put(msg)
        else:
            self._queue.put(msg)

    def log_run_meta(self, device: str = "", config: dict = None):
        self._put({
            "_kind": "run_meta",
            "start_time": time.time(),
            "device": device,
            "config_json": json.dumps(config or {}),
        })

    def log_epoch_summary(
        self,
        epoch: int,
        total_loss: float = None,
        policy_loss: float = None,
        value_loss: float = None,
        a_sfs_loss: float = None,
        plan_policy_loss: float = None,
        train_loss_avg: float = None,
        val_loss_avg: float = None,
        lr: float = None,
        grad_norm: float = None,
        top1_accuracy: float = None,
        top5_accuracy: float = None,
        duration_sec: float = None,
        num_samples: int = None,
    ):
        self._put({
            "_kind": "epoch_summary",
            "epoch": epoch,
            "total_loss": total_loss, "policy_loss": policy_loss, "value_loss": value_loss,
            "a_sfs_loss": a_sfs_loss, "plan_policy_loss": plan_policy_loss,
            "train_loss_avg": train_loss_avg, "val_loss_avg": val_loss_avg,
            "lr": lr, "grad_norm": grad_norm,
            "top1_accuracy": top1_accuracy, "top5_accuracy": top5_accuracy,
            "duration_sec": duration_sec, "num_samples": num_samples,
        })

    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float = None,
        policy_loss: float = None,
        value_loss: float = None,
        lr: float = None,
        grad_norm: float = None,
    ):
        self._put({
            "_kind": "batch_log",
            "epoch": epoch, "batch_idx": batch_idx,
            "total_loss": total_loss, "policy_loss": policy_loss, "value_loss": value_loss,
            "lr": lr, "grad_norm": grad_norm,
        })

    def log_gnn_internals(
        self,
        epoch: int,
        layer_name: str,
        weight_mean: float = None,
        weight_std: float = None,
        weight_min: float = None,
        weight_max: float = None,
        batch_idx: int = None,
        tensor_path: str = None,
    ):
        self._put({
            "_kind": "gnn_internals",
            "epoch": epoch, "batch_idx": batch_idx, "layer_name": layer_name,
            "weight_mean": weight_mean, "weight_std": weight_std, "weight_min": weight_min, "weight_max": weight_max,
            "tensor_path": tensor_path,
        })

    def log_mcts_stats(
        self,
        epoch: int = None,
        game_idx: int = None,
        move_idx: int = None,
        avg_depth: float = None,
        max_depth: int = None,
        nps: float = None,
        branching_factor: float = None,
        cutoff_rate: float = None,
        cache_hit_rate: float = None,
        puct_exploration_avg: float = None,
        puct_exploitation_avg: float = None,
        visit_histogram_json: str = None,
    ):
        self._put({
            "_kind": "mcts_stats",
            "epoch": epoch, "game_idx": game_idx, "move_idx": move_idx,
            "avg_depth": avg_depth, "max_depth": max_depth, "nps": nps,
            "branching_factor": branching_factor, "cutoff_rate": cutoff_rate, "cache_hit_rate": cache_hit_rate,
            "puct_exploration_avg": puct_exploration_avg, "puct_exploitation_avg": puct_exploitation_avg,
            "visit_histogram_json": visit_histogram_json,
        })

    def log_game_results(
        self,
        epoch: int = None,
        game_id: int = None,
        result: str = None,
        termination: str = None,
        color_white_elo_estimate: float = None,
        color_black_elo_estimate: float = None,
        avg_game_length: float = None,
        opening_eco: str = None,
        white_wins: int = None,
        black_wins: int = None,
        draws: int = None,
    ):
        self._put({
            "_kind": "game_results",
            "epoch": epoch, "game_id": game_id, "result": result, "termination": termination,
            "color_white_elo_estimate": color_white_elo_estimate, "color_black_elo_estimate": color_black_elo_estimate,
            "avg_game_length": avg_game_length, "opening_eco": opening_eco,
            "white_wins": white_wins, "black_wins": black_wins, "draws": draws,
        })

    def log_blunder_analysis(
        self,
        epoch: int = None,
        game_id: int = None,
        blunder_rate: float = None,
        centipawn_loss_vs_stockfish: float = None,
        centipawn_loss_vs_teacher: float = None,
    ):
        self._put({
            "_kind": "blunder_analysis",
            "epoch": epoch, "game_id": game_id, "blunder_rate": blunder_rate,
            "centipawn_loss_vs_stockfish": centipawn_loss_vs_stockfish, "centipawn_loss_vs_teacher": centipawn_loss_vs_teacher,
        })

    def log_endgame_stats(self, epoch: int = None, mate_in_x_errors_json: str = None):
        self._put({
            "_kind": "endgame_stats",
            "epoch": epoch, "mate_in_x_errors_json": mate_in_x_errors_json,
        })

    def log_hardware(
        self,
        timestamp: float = None,
        gpu_utilization_pct: float = None,
        vram_mb: float = None,
        gpu_temp_c: float = None,
        cpu_load_pct: float = None,
        ram_mb: float = None,
        disk_io_read: float = None,
        disk_io_write: float = None,
        positions_per_watt: float = None,
    ):
        self._put({
            "_kind": "hardware_snapshots",
            "timestamp": timestamp or time.time(),
            "gpu_utilization_pct": gpu_utilization_pct, "vram_mb": vram_mb, "gpu_temp_c": gpu_temp_c,
            "cpu_load_pct": cpu_load_pct, "ram_mb": ram_mb,
            "disk_io_read": disk_io_read, "disk_io_write": disk_io_write, "positions_per_watt": positions_per_watt,
        })

    def log_ticker(self, message: str):
        self._put({
            "_kind": "ticker_events",
            "timestamp": time.time(),
            "message": str(message),
        })

    def flush(self, timeout: float = 10.0):
        """Drain queue and wait for writer to finish (best-effort)."""
        self._flush_event.set()
        deadline = time.time() + timeout
        while time.time() < deadline and (self._queue.qsize() if hasattr(self._queue, 'qsize') else 0) > 0:
            time.sleep(0.05)

    def shutdown(self):
        self._stop.set()
        self._writer.join(timeout=5.0)

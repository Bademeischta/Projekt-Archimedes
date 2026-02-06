"""
Utility-Funktionen für Benchmark-Integration
"""

import json
import torch
from pathlib import Path
import psutil
import os


def is_colab_environment():
    """
    Erkennt, ob der Code in Google Colab läuft
    
    Returns:
        bool: True wenn in Colab-Umgebung
    """
    # Prüfe auf Colab-spezifische Umgebungsvariablen
    if os.environ.get('COLAB_GPU', None) is not None:
        return True
    
    # Prüfe auf typische Colab-Pfade
    if 'google.colab' in str(Path.cwd()) or '/content' in str(Path.cwd()):
        return True
    
    # Prüfe CPU-Kerne (Colab hat normalerweise 2 physische Kerne)
    cpu_physical = psutil.cpu_count(logical=False)
    if cpu_physical == 2:
        # Zusätzliche Prüfung: Colab hat typischerweise ~12GB RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if 10 <= ram_gb <= 15:
            return True
    
    return False


def load_benchmark_config(benchmark_file="benchmark_results.json"):
    """
    Lädt Benchmark-Konfiguration aus JSON-Datei
    
    Args:
        benchmark_file: Pfad zur Benchmark-JSON-Datei
        
    Returns:
        dict: Dictionary mit optimalen Parametern oder None falls Datei nicht existiert
    """
    benchmark_path = Path(benchmark_file)
    
    if not benchmark_path.exists():
        return None
    
    try:
        with open(benchmark_path, 'r') as f:
            data = json.load(f)
            return data.get("optimal_parameters", None)
    except Exception as e:
        print(f"Warnung: Konnte Benchmark-Datei nicht laden: {e}")
        return None


def get_auto_config(device=None, benchmark_file="benchmark_results.json"):
    """
    Ermittelt automatische Konfiguration basierend auf Benchmark oder aktueller Hardware
    
    Args:
        device: Optional device (wird überschrieben wenn Benchmark verfügbar)
        benchmark_file: Pfad zur Benchmark-JSON-Datei
        
    Returns:
        dict: Konfiguration mit batch_size, num_workers, pin_memory, etc.
    """
    # Versuche Benchmark zu laden
    benchmark_config = load_benchmark_config(benchmark_file)
    
    if benchmark_config:
        print(f"✓ Benchmark-Konfiguration geladen von {benchmark_file}")
        return benchmark_config
    
    # Fallback: Schnelle Auto-Detection ohne vollständigen Benchmark
    print("⚠ Keine Benchmark-Datei gefunden - verwende schnelle Auto-Detection")
    
    # Prüfe ob wir in Colab sind
    is_colab = is_colab_environment()
    if is_colab:
        print("🔵 Colab-Umgebung erkannt - verwende Colab-optimierte Parameter")
    
    config = {}
    
    # Device (als String speichern für JSON-Kompatibilität)
    if device is None:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if isinstance(device, torch.device):
            config["device"] = str(device)
        else:
            config["device"] = device
    
    # CPU Workers - Colab-optimiert
    cpu_physical = psutil.cpu_count(logical=False)
    
    if is_colab:
        # Colab hat nur 2 CPU-Kerne - verwende konservative Einstellungen
        # 1 Worker für DataLoader, 1 für Self-Play (oder Single-Worker Modus)
        config["num_workers"] = 1  # DataLoader Workers
        config["self_play_workers"] = 1  # Self-Play Workers (Single-Worker Modus)
    else:
        # Normale Umgebung - reserviere 25% oder min. 2 Kerne
        reserved_cores = max(2, int(cpu_physical * 0.25))
        available_cores = cpu_physical - reserved_cores
        
        config["num_workers"] = max(1, min(available_cores, 8))
        config["self_play_workers"] = max(1, available_cores)
    
    # GPU Batch-Size - Colab-optimiert
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / (1024**3)
        gpu_name = props.name.lower()
        
        if is_colab:
            # Colab-spezifische Optimierungen
            # T4 GPU: ~15GB Speicher, A100: ~40GB Speicher
            if 'a100' in gpu_name or memory_gb >= 30:
                config["batch_size"] = 64
                config["max_batch_size"] = 128
            elif 't4' in gpu_name or (memory_gb >= 12 and memory_gb < 30):
                # T4 ist die häufigste Colab GPU
                config["batch_size"] = 32
                config["max_batch_size"] = 64
            elif memory_gb >= 8:
                config["batch_size"] = 24
                config["max_batch_size"] = 48
            else:
                config["batch_size"] = 16
                config["max_batch_size"] = 32
        else:
            # Normale GPU-Schätzung
            if memory_gb >= 16:
                config["batch_size"] = 64
                config["max_batch_size"] = 128
            elif memory_gb >= 8:
                config["batch_size"] = 32
                config["max_batch_size"] = 64
            elif memory_gb >= 4:
                config["batch_size"] = 16
                config["max_batch_size"] = 32
            else:
                config["batch_size"] = 8
                config["max_batch_size"] = 16
        
        config["pin_memory"] = True
    else:
        cpu_physical = psutil.cpu_count(logical=False)
        if is_colab:
            # CPU-Modus in Colab - sehr konservativ
            config["batch_size"] = 8
            config["max_batch_size"] = 16
        else:
            reserved_cores = max(2, int(cpu_physical * 0.25))
            available_cores = cpu_physical - reserved_cores
            config["batch_size"] = min(32, max(8, int(available_cores / 2)))
            config["max_batch_size"] = config["batch_size"]
        config["pin_memory"] = False
    
    # RAM-basierte Buffer-Größe - Colab-optimiert
    ram = psutil.virtual_memory()
    available_ram_gb = ram.available / (1024**3)
    
    if is_colab:
        # Colab hat begrenzten RAM (~12GB) - konservativere Buffer-Größe
        # Reserve mehr RAM für System (3GB statt 2GB)
        training_ram_gb = max(0, available_ram_gb - 3)
        # Reduzierte Buffer-Größe für Colab
        config["replay_buffer_size"] = min(20000, int(training_ram_gb * 1024 * 1024 / 15))
    else:
        training_ram_gb = max(0, available_ram_gb - 2)
        config["replay_buffer_size"] = min(50000, int(training_ram_gb * 1024 * 1024 / 10))
    
    config["prefetch_factor"] = 2 if config["num_workers"] > 0 else None
    config["persistent_workers"] = config["num_workers"] > 0
    
    return config


def apply_auto_config(args, benchmark_file="benchmark_results.json"):
    """
    Wendet automatische Konfiguration auf Argumente an
    
    Args:
        args: argparse.Namespace mit Trainings-Argumenten
        benchmark_file: Pfad zur Benchmark-JSON-Datei
        
    Returns:
        argparse.Namespace: Modifizierte Argumente
    """
    # Device als String für get_auto_config
    device_str = None
    if hasattr(args, 'device') and args.device is not None:
        if isinstance(args.device, torch.device):
            device_str = str(args.device)
        else:
            device_str = args.device
    
    auto_config = get_auto_config(device_str, benchmark_file)
    
    # Überschreibe nur wenn nicht explizit gesetzt
    if not hasattr(args, 'batch_size') or args.batch_size is None:
        args.batch_size = auto_config.get("batch_size", 32)
    
    if not hasattr(args, 'device') or args.device is None:
        device_from_config = auto_config.get("device", "cpu")
        if isinstance(device_from_config, str):
            args.device = device_from_config  # Wird später zu torch.device konvertiert
        else:
            args.device = device_from_config
    
    if not hasattr(args, 'num_workers') or args.num_workers is None:
        args.num_workers = auto_config.get("num_workers", 2)
    
    if not hasattr(args, 'pin_memory') or args.pin_memory is None:
        args.pin_memory = auto_config.get("pin_memory", False)
    
    if not hasattr(args, 'replay_buffer_size') or args.replay_buffer_size is None:
        args.replay_buffer_size = auto_config.get("replay_buffer_size", 10000)
    
    return args

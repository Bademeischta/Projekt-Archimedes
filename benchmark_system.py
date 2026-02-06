"""
System Benchmark Script für Archimedes Training
Testet Hardware und schlägt optimale Trainingsparameter vor.
"""

import torch
import psutil
import time
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.archimedes.model import TPN, SAN, PlanToMoveMapper


def benchmark_cpu():
    """Benchmark CPU Performance"""
    print("\n=== CPU Benchmark ===")
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    
    print(f"CPU Kerne (logisch): {cpu_count}")
    print(f"CPU Kerne (physisch): {cpu_count_physical}")
    
    # CPU Geschwindigkeit testen
    print("Teste CPU Geschwindigkeit...")
    start = time.time()
    result = sum(i * i for i in range(10000000))
    cpu_time = time.time() - start
    print(f"CPU Benchmark Zeit: {cpu_time:.4f} Sekunden")
    
    # CPU-Auslastung prüfen
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"Aktuelle CPU-Auslastung: {cpu_percent:.1f}%")
    
    return {
        "cores_logical": cpu_count,
        "cores_physical": cpu_count_physical,
        "benchmark_time": cpu_time,
        "current_usage": cpu_percent
    }


def benchmark_ram():
    """Benchmark RAM"""
    print("\n=== RAM Benchmark ===")
    ram = psutil.virtual_memory()
    total_gb = ram.total / (1024**3)
    available_gb = ram.available / (1024**3)
    used_gb = ram.used / (1024**3)
    percent_used = ram.percent
    
    print(f"Gesamter RAM: {total_gb:.2f} GB")
    print(f"Verfügbarer RAM: {available_gb:.2f} GB")
    print(f"Verwendeter RAM: {used_gb:.2f} GB ({percent_used:.1f}%)")
    
    # RAM-Geschwindigkeit testen
    print("Teste RAM Geschwindigkeit...")
    size_mb = 100
    data = np.random.rand(size_mb * 1024 * 1024 // 8).astype(np.float64)
    
    start = time.time()
    _ = np.sum(data)
    ram_time = time.time() - start
    print(f"RAM Benchmark Zeit (100MB): {ram_time:.4f} Sekunden")
    
    return {
        "total_gb": total_gb,
        "available_gb": available_gb,
        "used_gb": used_gb,
        "percent_used": percent_used,
        "benchmark_time": ram_time
    }


def benchmark_gpu():
    """Benchmark GPU Performance"""
    print("\n=== GPU Benchmark ===")
    
    if not torch.cuda.is_available():
        print("Keine GPU verfügbar")
        return None
    
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    
    results = {}
    
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / (1024**3)
        
        print(f"  Name: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Gesamter Speicher: {total_memory_gb:.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # GPU-Speicher testen
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        
        # Teste verschiedene Batch-Sizes
        print("  Teste GPU-Performance mit verschiedenen Batch-Sizes...")
        batch_sizes = [8, 16, 32, 64, 128, 256]
        optimal_batch_size = 32
        max_batch_size = 32
        
        for bs in batch_sizes:
            try:
                # Teste mit TPN Modell
                model = TPN().to(device)
                dummy_input = torch.randn(bs, 22, 8, 8, device=device)
                
                # Warmup
                for _ in range(3):
                    _ = model(dummy_input)
                
                torch.cuda.synchronize()
                start = time.time()
                
                for _ in range(10):
                    _ = model(dummy_input)
                
                torch.cuda.synchronize()
                elapsed = (time.time() - start) / 10
                
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)
                
                print(f"    Batch-Size {bs:3d}: {elapsed*1000:.2f}ms/Batch, "
                      f"Speicher: {memory_allocated:.1f}MB / {memory_reserved:.1f}MB")
                
                if memory_reserved < total_memory_gb * 1024 * 0.8:  # Max 80% GPU-Speicher
                    max_batch_size = bs
                    optimal_batch_size = bs
                
                del model, dummy_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    Batch-Size {bs:3d}: OUT OF MEMORY")
                    break
                else:
                    raise
        
        # Teste Training-Loop Performance
        print("  Teste Training-Loop Performance...")
        model = TPN().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        dummy_input = torch.randn(optimal_batch_size, 22, 8, 8, device=device)
        dummy_target = torch.randint(0, 4672, (optimal_batch_size,), device=device)
        
        # Warmup
        for _ in range(5):
            output, _ = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        start = time.time()
        
        iterations = 20
        for _ in range(iterations):
            output, _ = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        training_time = (time.time() - start) / iterations
        
        print(f"  Training-Loop Zeit: {training_time*1000:.2f}ms pro Iteration")
        
        del model, optimizer, dummy_input, dummy_target
        torch.cuda.empty_cache()
        
        results[f"gpu_{i}"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": total_memory_gb,
            "multiprocessors": props.multi_processor_count,
            "optimal_batch_size": optimal_batch_size,
            "max_batch_size": max_batch_size,
            "training_time_ms": training_time * 1000
        }
    
    return results


def calculate_optimal_parameters(cpu_info, ram_info, gpu_info):
    """Berechnet optimale Trainingsparameter basierend auf Benchmark-Ergebnissen"""
    print("\n=== Optimale Parameter Berechnung ===")
    
    params = {}
    
    # CPU-basierte Parameter
    cpu_cores = cpu_info["cores_logical"]
    cpu_physical = cpu_info["cores_physical"]
    
    # Lasse 2 Kerne frei für System (oder 25%, je nachdem was größer ist)
    reserved_cores = max(2, int(cpu_physical * 0.25))
    available_cores = cpu_physical - reserved_cores
    
    params["num_workers"] = max(1, min(available_cores, 8))  # Max 8 Worker für DataLoader
    params["self_play_workers"] = max(1, available_cores)  # Mehr Worker für Self-Play OK
    
    print(f"Verfügbare CPU-Kerne: {available_cores} (von {cpu_physical}, {reserved_cores} reserviert)")
    print(f"Empfohlene DataLoader Workers: {params['num_workers']}")
    print(f"Empfohlene Self-Play Workers: {params['self_play_workers']}")
    
    # RAM-basierte Parameter
    available_ram_gb = ram_info["available_gb"]
    
    # Reserve 2GB RAM für System
    training_ram_gb = max(0, available_ram_gb - 2)
    
    # Geschätzte RAM-Nutzung pro Batch (sehr grob)
    # TPN: ~22*8*8*4 bytes pro Sample = ~5.5KB, Batch-Size 32 = ~180KB
    # Plus Overhead für PyTorch, etc.
    params["replay_buffer_size"] = min(50000, int(training_ram_gb * 1024 * 1024 / 10))  # Sehr konservativ
    
    print(f"Verfügbarer RAM für Training: {training_ram_gb:.2f} GB")
    print(f"Empfohlene Replay Buffer Größe: {params['replay_buffer_size']}")
    
    # GPU-basierte Parameter
    if gpu_info:
        gpu_0 = gpu_info.get("gpu_0", {})
        if gpu_0:
            params["batch_size"] = gpu_0.get("optimal_batch_size", 32)
            params["max_batch_size"] = gpu_0.get("max_batch_size", 32)
            params["pin_memory"] = True  # Nur wenn GPU verfügbar
            params["device"] = "cuda"
            
            print(f"GPU verfügbar: {gpu_0.get('name', 'Unknown')}")
            print(f"Empfohlene Batch-Size: {params['batch_size']}")
            print(f"Maximale Batch-Size: {params['max_batch_size']}")
            print(f"Pin Memory: {params['pin_memory']}")
    else:
        params["batch_size"] = min(32, max(8, int(available_cores / 2)))
        params["max_batch_size"] = params["batch_size"]
        params["pin_memory"] = False
        params["device"] = "cpu"
        
        print("Keine GPU verfügbar - verwende CPU")
        print(f"Empfohlene Batch-Size (CPU): {params['batch_size']}")
    
    # Zusätzliche Optimierungen
    params["prefetch_factor"] = 2 if params["num_workers"] > 0 else None
    params["persistent_workers"] = params["num_workers"] > 0
    
    return params


def save_benchmark_results(cpu_info, ram_info, gpu_info, params, output_file):
    """Speichert Benchmark-Ergebnisse in JSON"""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu_info,
        "ram": ram_info,
        "gpu": gpu_info,
        "optimal_parameters": params
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark-Ergebnisse gespeichert in: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark System für Archimedes Training")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Ausgabedatei für Benchmark-Ergebnisse")
    parser.add_argument("--skip-gpu-test", action="store_true",
                       help="Überspringe GPU-Tests (schneller)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Archimedes System Benchmark")
    print("=" * 60)
    
    # CPU Benchmark
    cpu_info = benchmark_cpu()
    
    # RAM Benchmark
    ram_info = benchmark_ram()
    
    # GPU Benchmark
    gpu_info = None
    if not args.skip_gpu_test:
        gpu_info = benchmark_gpu()
    else:
        if torch.cuda.is_available():
            print("\n=== GPU Info (ohne Benchmark) ===")
            props = torch.cuda.get_device_properties(0)
            gpu_info = {
                "gpu_0": {
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}"
                }
            }
            print(f"GPU: {props.name}")
            print(f"Speicher: {props.total_memory / (1024**3):.2f} GB")
    
    # Berechne optimale Parameter
    params = calculate_optimal_parameters(cpu_info, ram_info, gpu_info)
    
    # Zeige Zusammenfassung
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG - Empfohlene Parameter")
    print("=" * 60)
    print(f"Device: {params['device']}")
    print(f"Batch-Size: {params['batch_size']}")
    print(f"DataLoader Workers: {params['num_workers']}")
    print(f"Self-Play Workers: {params['self_play_workers']}")
    print(f"Pin Memory: {params['pin_memory']}")
    print(f"Replay Buffer Größe: {params['replay_buffer_size']}")
    print("=" * 60)
    
    # Speichere Ergebnisse
    output_path = Path(args.output)
    save_benchmark_results(cpu_info, ram_info, gpu_info, params, output_path)
    
    print("\nVerwende diese Parameter in deinen Trainings-Scripts:")
    print(f"  --batch-size {params['batch_size']}")
    print(f"  --num-workers {params['self_play_workers']}  # für train_end_to_end.py")
    print(f"  --device {params['device']}")
    print("\nOder verwende --auto-config in den Trainings-Scripts!")


if __name__ == "__main__":
    main()

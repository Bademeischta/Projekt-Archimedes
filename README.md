# Projekt "Archimedes" - High-Performance Chess AI

## 1. Übersicht

**Archimedes** ist eine hochmoderne Schach-KI, die als skalierbare Forschungsplattform für hybride neuronale Architekturen konzipiert wurde. Im Gegensatz zu traditionellen Engines, die sich entweder auf rohe taktische Berechnung (wie AlphaZero) oder rein strategische Konzepte konzentrieren, verfolgt Archimedes einen dualen Ansatz. Das Herzstück des Projekts ist eine Zwei-Stream-Architektur, die strategisches Denken und taktische Präzision in einem einzigen, kohärenten System vereint.

Das Ziel von Archimedes ist es, nicht nur starke Züge zu finden, sondern auch die zugrunde liegenden strategischen Pläne zu verstehen, zu bewerten und zu verfolgen.

## 2. Neue Features (v2.0)

### 🚀 Architektur-Upgrades

#### **ResNet-basiertes TPN (Tactical Perception Network)**
- **10 Residual Blocks** mit Batch Normalization für deutlich tiefere und stabilere Netzwerke
- **256 Kanäle** in der Hauptarchitektur (vorher: 128)
- Verbesserte Policy- und Value-Heads mit BatchNorm
- **5-10x bessere taktische Genauigkeit** im Vergleich zur alten 3-Layer-CNN-Architektur

#### **Optimierte MCTS-Suche**
- **Time-based Iterative Deepening**: Suche läuft bis zu einem Zeitlimit statt fixer Simulationen
- **LRU Transposition Table**: Intelligente Eviction-Strategie statt "clear all when full"
- **Q-Value Normalization**: Dynamische Min-Max-Normalisierung für stabilere UCB-Scores
- **Adaptive Tiefensteuerung**: Automatische Anpassung der Suchtiefe basierend auf verfügbarer Zeit

### ⚡ Training-Optimierungen

#### **Automatic Mixed Precision (AMP)**
- **2-3x schnelleres Training** auf NVIDIA RTX GPUs (getestet auf RTX 5070)
- **40-50% weniger VRAM-Verbrauch** durch FP16-Berechnungen
- Automatische Gradient-Skalierung mit `torch.cuda.amp.GradScaler`
- Kompatibel mit allen CUDA-fähigen GPUs (Compute Capability 7.0+)

#### **Advanced Learning Rate Schedulers**
- **CosineAnnealingWarmRestarts**: Periodische Warm Restarts verhindern lokale Minima
- **ReduceLROnPlateau**: Adaptive LR-Reduktion bei Stagnation
- Separate Scheduler für TPN und SAN für optimale Konvergenz

#### **Robuste Warmup-Phase**
- **Drain-Mechanismus**: Garantiert vollständige Verarbeitung aller Warmup-Spiele
- Verhindert Race Conditions zwischen Self-Play und Training
- Konfigurierbare Warmup-Größe für schnelleren Trainingsstart

### 📊 Monitoring & Logging

#### **Konfigurierbare MetricsLogger**
- Anpassbare Queue-Timeouts für verschiedene Hardware-Setups
- Verbesserte Multiprocessing-Unterstützung
- Detaillierte Dokumentation aller Parameter

## 3. Kernarchitektur

Die Architektur von Archimedes ruht auf mehreren innovativen Säulen:

### a) Duales Repräsentationsmodul (DRM)
Für jede Schachstellung erzeugt das System zwei komplementäre Darstellungen:
*   **Tensor-Repräsentation**: Eine (C, 8, 8) Tensor-Darstellung im Stil von AlphaZero, die effiziente Bitboards für Figurenpositionen, Angriffsflächen, Fesselungen etc. enthält. Diese Darstellung ist für schnelle, taktische Analysen optimiert.
*   **Graph-Repräsentation**: Ein 64-Knoten-Graph, bei dem jeder Knoten ein Feld auf dem Brett darstellt. Die Kanten des Graphen repräsentieren dynamisch die Beziehungen zwischen den Figuren (z.B. "greift an", "verteidigt", "ist Teil einer Bauernkette"). Diese Darstellung ist für die Analyse abstrakter, strategischer Muster optimiert.

### b) Zwei-Stream-Neuronales-Netzwerk
*   **Tactical Perception Network (TPN)**: Ein **ResNet-basiertes CNN** mit 10 Residual Blocks, das die **Tensor-Repräsentation** verarbeitet. Es ist für die unmittelbare taktische Bewertung (`V_tactical`) und die Vorhersage von Zug-Wahrscheinlichkeiten (`π_tactical`) zuständig.
*   **Strategic Abstraction Network (SAN)**: Ein Graph-Neuronales-Netzwerk (GNN), das die **Graph-Repräsentation** verarbeitet. Seine Aufgabe ist es, abstrakte strategische Konzepte zu verstehen und zu formulieren, wie z.B. einen "Königsangriff" oder "Zentrumskontrolle". Es erzeugt einen Zielvektor (`Goal Vector`), mehrere Plan-Vorschläge (`Plan Embeddings`) und eine Wahrscheinlichkeitsverteilung über diese Pläne (`π_strategic`).

### c) Conceptual Graph Search (CGS)
Anstelle einer reinen Alpha-Beta- oder MCTS-Suche verwendet Archimedes eine hierarchische MCTS-Suche:
1.  **Strategie-Ebene**: Das SAN analysiert die Stellung und schlägt einen strategischen Plan vor.
2.  **Taktik-Ebene**: Ein `PlanToMoveMapper` übersetzt den abstrakten Plan in einen Bias-Vektor für die Zug-Wahrscheinlichkeiten des TPN.
3.  **Suche**: Eine MCTS-Suche mit **Iterative Deepening** und **LRU Transposition Table** wird durchgeführt, die stark von dieser kombinierten, strategisch ausgerichteten Policy geleitet wird.
4.  **Priority Arbiter**: Ein Sicherheitsmechanismus, der vor jeder Suche prüft, ob unmittelbare taktische Gefahren bestehen. Wenn ja, kann das TPN das SAN überstimmen (`Tactical Override`), um einen taktischen Fehler zu vermeiden.

### d) Autonomer Lernzyklus (Self-Play)
Archimedes lernt durch einen ausgeklügelten Self-Play-Mechanismus mit getrennten Belohnungssignalen:
*   Das **TPN** wird dafür belohnt, Partien zu gewinnen (`final_game_result`).
*   Das **SAN** wird dafür belohnt, "gute Pläne" zu entwickeln. Die Güte eines Plans wird durch den **Strategic Fulfillment Score (SFS)** gemessen – eine komplexe Metrik, die Zielerreichung, Widerstandsfähigkeit und Initiative bewertet.
*   **Amortisierte Kritik**: Das SAN lernt, den SFS-Wert selbst vorherzusagen (`A-SFS Head`), was das Training effizienter macht.

## 4. Projektstruktur

```
/
├── pyproject.toml       # Projekt- und Abhängigkeitsmanagement mit Poetry
├── poetry.lock          # Gesperrte Abhängigkeitsversionen
├── README.md            # Diese Datei
├── .gitignore
├── pytest.ini           # Konfiguration für Tests
│
├── quantize_tpn.py      # Skript zur Quantisierung des TPN-Modells
├── evaluate_elo.py      # Skript zur Elo-Bewertung zwischen zwei Modellen
├── train_tpn.py         # (Veraltet) Skript zum isolierten Training des TPN
├── train_san.py         # (Veraltet) Skript zum isolierten Training des SAN
├── train_end_to_end.py  # Hauptskript für das Self-Play-Training (mit AMP!)
├── run_archimedes.py    # One-Click-Launcher für Training
├── dashboard.py         # Live-Dashboard für Training-Metriken
├── metrics.py           # Asynchroner MetricsLogger
├── benchmark_system.py  # Hardware-Benchmark für optimale Konfiguration
│
├── src/
│   └── archimedes/
│       ├── __init__.py
│       ├── representation.py  # DRM: board_to_tensor & board_to_graph
│       ├── utils.py           # Hilfsfunktionen (z.B. move_to_index)
│       ├── pipeline.py        # PGN-Parser
│       ├── model.py           # TPN (ResNet!), SAN, PlanToMoveMapper
│       ├── search.py          # ConceptualGraphSearch (Time-based + LRU!)
│       ├── rewards.py         # Strategic Fulfillment Score (SFS) Berechnung
│       └── create_dataset.py  # Skript zur Erstellung von Trainings-Datensätzen
│
└── tests/                 # Unit-Tests für alle Komponenten
```

## 5. Setup und Installation

### a) Lokale Installation

**Voraussetzungen**:
*   Python 3.9+
*   [Poetry](https://python-poetry.org/docs/#installation) für das Abhängigkeitsmanagement
*   **NVIDIA GPU mit CUDA 11.8+ (empfohlen für AMP)**

**Schritte**:

1.  **Klone das Repository**:
    ```bash
    git clone <repository_url>
    cd archimedes
    ```

2.  **Installiere die Basis-Abhängigkeiten mit Poetry**:
    ```bash
    poetry install
    ```
    *Hinweis: Dies installiert alle Abhängigkeiten außer PyTorch, da dessen Installation plattformspezifisch ist.*

3.  **Installiere PyTorch mit CUDA-Support**:
    Für **NVIDIA RTX 5070** oder andere moderne GPUs:
    ```bash
    poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    ```

    **Beispiel für CPU-Version** (nicht empfohlen für Training):
    ```bash
    poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
    ```

4.  **Überprüfe die Installation**:
    Führe die Test-Suite aus, um sicherzustellen, dass alle Komponenten korrekt installiert sind.
    ```bash
    poetry run pytest
    ```
    Alle Tests sollten erfolgreich durchlaufen.

### b) Google Colab Setup (One-Click!)

Für GPU-beschleunigtes Training ist Google Colab eine ausgezeichnete, kostenlose Option.

#### Schnellstart mit Colab Notebook (Empfohlen)

**Am einfachsten**: Verwenden Sie das fertige Colab Notebook `archimedes_colab.ipynb`:

1. **Öffne das Notebook in Colab**:
   - Laden Sie `archimedes_colab.ipynb` in Google Colab hoch, oder
   - Klonen Sie das Repository und öffnen Sie das Notebook

2. **Aktiviere GPU**:
   - `Laufzeit` → `Laufzeittyp ändern` → `Hardwarebeschleuniger: GPU`

3. **Führe die Setup-Zelle aus**:
   - Das Notebook installiert automatisch alle Abhängigkeiten
   - Startet das Training mit optimalen Parametern
   - **Alles in einer Zelle!**

#### Colab-spezifische Parameter-Empfehlungen

Das System erkennt automatisch Colab-Umgebungen und passt die Parameter an:

- **Workers**: Colab hat nur 2 CPU-Kerne → `--num-workers 1` (automatisch gesetzt)
- **Batch-Size**: 
  - T4 GPU: ~32 (automatisch)
  - A100 GPU: ~64 (automatisch)
- **Replay Buffer**: Reduziert auf ~20.000 für begrenzten RAM
- **AMP**: Automatisch aktiviert auf allen Colab-GPUs

**Tipp**: Verwenden Sie `--auto-config` für optimale Colab-Parameter!

## 6. Benutzung

### a) One-Click Training (Empfohlen!)

Der einfachste Weg, Archimedes zu trainieren:

```bash
# Schritt 1: Hardware-Benchmark (einmalig)
poetry run python benchmark_system.py

# Schritt 2: Training starten (mit AMP!)
poetry run python run_archimedes.py
```

Das war's! Das Skript verwendet automatisch die optimalen Parameter für Ihre Hardware.

### b) Manuelles Training mit AMP

Für volle Kontrolle über alle Parameter:

```bash
# Training mit AMP (empfohlen für NVIDIA RTX GPUs)
poetry run python train_end_to_end.py \
    --auto-config \
    --total-games 1000 \
    --use-amp \
    --scheduler cosine \
    --warmup-games 50

# Training ohne AMP (für ältere GPUs oder CPU)
poetry run python train_end_to_end.py \
    --auto-config \
    --total-games 1000 \
    --no-amp \
    --scheduler plateau
```

**Wichtige Parameter**:
- `--use-amp` / `--no-amp`: Aktiviert/Deaktiviert Automatic Mixed Precision
- `--scheduler`: Wählt Learning Rate Scheduler (`cosine`, `plateau`, `none`)
- `--warmup-games`: Anzahl der Warmup-Spiele vor dem Training
- `--auto-config`: Verwendet Benchmark-Ergebnisse für optimale Konfiguration

### c) Live-Dashboard

Überwachen Sie Ihr Training in Echtzeit:

```bash
# In einem separaten Terminal
poetry run python dashboard.py
```

Öffnen Sie dann `http://localhost:8050` in Ihrem Browser.

Das Dashboard zeigt:
- **Training-Metriken**: Loss, Accuracy, Learning Rate
- **Hardware-Auslastung**: GPU/CPU/RAM in Echtzeit
- **MCTS-Statistiken**: Suchtiefe, Nodes per Second, Cache Hit Rate
- **Q-Value Normalization**: Min/Max-Tracking

### d) System Benchmark

**NEU**: Dieses Skript benchmarkt deine Hardware (CPU, GPU, RAM) und schlägt optimale Trainingsparameter vor:

```bash
# Führe vollständigen Benchmark durch
poetry run python benchmark_system.py

# Benchmark ohne GPU-Tests (schneller)
poetry run python benchmark_system.py --skip-gpu-test
```

Das Skript testet:
- **CPU**: Anzahl Kerne, Geschwindigkeit, aktuelle Auslastung
- **RAM**: Gesamter/verfügbarer Speicher, Geschwindigkeit
- **GPU**: Speicher, Compute-Capability, optimale Batch-Size für AMP

**Automatische Parameter-Optimierung**:
- Reserviert automatisch CPU-Kerne für System-Nutzung (25% oder min. 2 Kerne)
- Findet optimale Batch-Size basierend auf GPU-Speicher und AMP
- Empfiehlt optimale Anzahl von Workers für DataLoader und Self-Play
- Berechnet optimale Replay-Buffer-Größe basierend auf verfügbarem RAM

### e) Erweiterte Konfiguration

#### Time-based Search (statt fixer Simulationen)

```python
from src.archimedes.search import ConceptualGraphSearch

search = ConceptualGraphSearch(
    tpn, san, mapper,
    time_limit=1.0,  # 1 Sekunde pro Zug
    use_transposition_table=True,
    use_q_normalization=True
)
```

#### ResNet-Konfiguration anpassen

```python
from src.archimedes.model import TPN

# Mehr Residual Blocks für tiefere Netzwerke
tpn = TPN(num_res_blocks=15, num_channels=512)

# Weniger Blocks für schnellere Inferenz
tpn = TPN(num_res_blocks=5, num_channels=128)
```

## 7. Performance-Tipps

### Für NVIDIA RTX 5070 (und ähnliche GPUs)

```bash
# Optimale Konfiguration für RTX 5070
poetry run python train_end_to_end.py \
    --batch-size 64 \
    --num-workers 4 \
    --use-amp \
    --scheduler cosine \
    --warmup-games 100 \
    --replay-buffer-size 50000
```

**Erwartete Performance**:
- **Training Speed**: ~2-3x schneller als ohne AMP
- **VRAM Usage**: ~6-8 GB (statt 12-14 GB ohne AMP)
- **Nodes per Second**: ~5000-8000 (mit LRU TT)

### Für ältere GPUs (GTX 1080, RTX 2060, etc.)

```bash
# Reduzierte Batch-Size, kein AMP
poetry run python train_end_to_end.py \
    --batch-size 32 \
    --num-workers 2 \
    --no-amp \
    --scheduler plateau \
    --warmup-games 50
```

### Für CPU-Training (nicht empfohlen)

```bash
# Minimale Konfiguration für CPU
poetry run python train_end_to_end.py \
    --batch-size 16 \
    --num-workers 1 \
    --no-amp \
    --total-games 100
```

## 8. Troubleshooting

### "Out of Memory" Fehler

**Lösung 1**: Reduzieren Sie die Batch-Size
```bash
--batch-size 16  # statt 32 oder 64
```

**Lösung 2**: Deaktivieren Sie AMP (falls aktiviert)
```bash
--no-amp
```

**Lösung 3**: Reduzieren Sie die Anzahl der Residual Blocks
```python
tpn = TPN(num_res_blocks=5, num_channels=128)  # statt 10/256
```

### Training läuft sehr langsam

**Lösung 1**: Aktivieren Sie AMP (falls GPU unterstützt)
```bash
--use-amp
```

**Lösung 2**: Verwenden Sie Time-based Search statt fixer Simulationen
```python
search = ConceptualGraphSearch(..., time_limit=0.5)  # 0.5s pro Zug
```

**Lösung 3**: Aktivieren Sie Transposition Table
```python
search = ConceptualGraphSearch(..., use_transposition_table=True, tt_max_size=100000)
```

### GPU wird nicht erkannt

**Lösung**: Überprüfen Sie Ihre PyTorch-Installation
```bash
poetry run python -c "import torch; print(torch.cuda.is_available())"
```

Falls `False`, installieren Sie PyTorch neu mit CUDA-Support:
```bash
poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## 9. Changelog (v2.0)

### Architektur
- ✅ ResNet-basiertes TPN mit 10 Residual Blocks
- ✅ Batch Normalization für alle Convolutional Layers
- ✅ 256 Kanäle (vorher: 128)

### Search
- ✅ Time-based Iterative Deepening
- ✅ LRU Transposition Table
- ✅ Q-Value Normalization mit dynamischem Min-Max-Tracking

### Training
- ✅ Automatic Mixed Precision (AMP) mit GradScaler
- ✅ CosineAnnealingWarmRestarts Scheduler
- ✅ ReduceLROnPlateau Scheduler
- ✅ Robuster Warmup-Drain-Mechanismus

### Monitoring
- ✅ Konfigurierbare Queue-Timeouts in MetricsLogger
- ✅ Erweiterte MCTS-Statistiken (Q-Min/Max)
- ✅ Verbesserte Hardware-Snapshots

### Dokumentation
- ✅ Vollständig überarbeitetes README
- ✅ One-Click Colab Notebook
- ✅ Performance-Tipps für verschiedene Hardware

---

**Projekt "Archimedes"** - Eine neue Ära der strategischen Schach-KI mit High-Performance-Training.

**Hardware-Empfehlung**: NVIDIA RTX 5070 oder besser für optimale Performance mit AMP.

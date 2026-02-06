# Projekt "Archimedes"

## 1. Übersicht

**Archimedes** ist eine hochmoderne Schach-KI, die als skalierbare Forschungsplattform für hybride neuronale Architekturen konzipiert wurde. Im Gegensatz zu traditionellen Engines, die sich entweder auf rohe taktische Berechnung (wie AlphaZero) oder rein strategische Konzepte konzentrieren, verfolgt Archimedes einen dualen Ansatz. Das Herzstück des Projekts ist eine Zwei-Stream-Architektur, die strategisches Denken und taktische Präzision in einem einzigen, kohärenten System vereint.

Das Ziel von Archimedes ist es, nicht nur starke Züge zu finden, sondern auch die zugrunde liegenden strategischen Pläne zu verstehen, zu bewerten und zu verfolgen.

## 2. Kernarchitektur

Die Architektur von Archimedes ruht auf mehreren innovativen Säulen:

### a) Duales Repräsentationsmodul (DRM)
Für jede Schachstellung erzeugt das System zwei komplementäre Darstellungen:
*   **Tensor-Repräsentation**: Eine (C, 8, 8) Tensor-Darstellung im Stil von AlphaZero, die effiziente Bitboards für Figurenpositionen, Angriffsflächen, Fesselungen etc. enthält. Diese Darstellung ist für schnelle, taktische Analysen optimiert.
*   **Graph-Repräsentation**: Ein 64-Knoten-Graph, bei dem jeder Knoten ein Feld auf dem Brett darstellt. Die Kanten des Graphen repräsentieren dynamisch die Beziehungen zwischen den Figuren (z.B. "greift an", "verteidigt", "ist Teil einer Bauernkette"). Diese Darstellung ist für die Analyse abstrakter, strategischer Muster optimiert.

### b) Zwei-Stream-Neuronales-Netzwerk
*   **Tactical Perception Network (TPN)**: Ein schnelles, CNN-basiertes Netzwerk, das die **Tensor-Repräsentation** verarbeitet. Es ist für die unmittelbare taktische Bewertung (`V_tactical`) und die Vorhersage von Zug-Wahrscheinlichkeiten (`π_tactical`) zuständig.
*   **Strategic Abstraction Network (SAN)**: Ein Graph-Neuronales-Netzwerk (GNN), das die **Graph-Repräsentation** verarbeitet. Seine Aufgabe ist es, abstrakte strategische Konzepte zu verstehen und zu formulieren, wie z.B. einen "Königsangriff" oder "Zentrumskontrolle". Es erzeugt einen Zielvektor (`Goal Vector`), mehrere Plan-Vorschläge (`Plan Embeddings`) und eine Wahrscheinlichkeitsverteilung über diese Pläne (`π_strategic`).

### c) Conceptual Graph Search (CGS)
Anstelle einer reinen Alpha-Beta- oder MCTS-Suche verwendet Archimedes eine hierarchische MCTS-Suche:
1.  **Strategie-Ebene**: Das SAN analysiert die Stellung und schlägt einen strategischen Plan vor.
2.  **Taktik-Ebene**: Ein `PlanToMoveMapper` übersetzt den abstrakten Plan in einen Bias-Vektor für die Zug-Wahrscheinlichkeiten des TPN.
3.  **Suche**: Eine MCTS-Suche wird durchgeführt, die stark von dieser kombinierten, strategisch ausgerichteten Policy geleitet wird.
4.  **Priority Arbiter**: Ein Sicherheitsmechanismus, der vor jeder Suche prüft, ob unmittelbare taktische Gefahren bestehen. Wenn ja, kann das TPN das SAN überstimmen (`Tactical Override`), um einen taktischen Fehler zu vermeiden.

### d) Autonomer Lernzyklus (Self-Play)
Archimedes lernt durch einen ausgeklügelten Self-Play-Mechanismus mit getrennten Belohnungssignalen:
*   Das **TPN** wird dafür belohnt, Partien zu gewinnen (`final_game_result`).
*   Das **SAN** wird dafür belohnt, "gute Pläne" zu entwickeln. Die Güte eines Plans wird durch den **Strategic Fulfillment Score (SFS)** gemessen – eine komplexe Metrik, die Zielerreichung, Widerstandsfähigkeit und Initiative bewertet.
*   **Amortisierte Kritik**: Das SAN lernt, den SFS-Wert selbst vorherzusagen (`A-SFS Head`), was das Training effizienter macht.

## 3. Projektstruktur

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
├── train_end_to_end.py  # Hauptskript für das Self-Play-Training
│
├── src/
│   └── archimedes/
│       ├── __init__.py
│       ├── representation.py  # DRM: board_to_tensor & board_to_graph
│       ├── utils.py           # Hilfsfunktionen (z.B. move_to_index)
│       ├── pipeline.py        # PGN-Parser
│       ├── model.py           # TPN, SAN, PlanToMoveMapper Klassen
│       ├── search.py          # ConceptualGraphSearch (MCTS & Priority Arbiter)
│       ├── rewards.py         # Strategic Fulfillment Score (SFS) Berechnung
│       └── create_dataset.py  # Skript zur Erstellung von Trainings-Datensätzen
│
└── tests/                 # Unit-Tests für alle Komponenten
```

## 4. Setup und Installation

### a) Lokale Installation

**Voraussetzungen**:
*   Python 3.9+
*   [Poetry](https://python-poetry.org/docs/#installation) für das Abhängigkeitsmanagement

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

3.  **Installiere PyTorch**:
    PyTorch muss manuell innerhalb der von Poetry erstellten virtuellen Umgebung installiert werden. Wähle den Befehl, der zu deinem System passt (CPU, CUDA, etc.). Die offizielle Anleitung findest du [hier](https://pytorch.org/get-started/locally/).

    **Beispiel für CPU-Version**:
    ```bash
    poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
    ```

    **Beispiel für CUDA 11.8**:
    ```bash
    poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Überprüfe die Installation**:
    Führe die Test-Suite aus, um sicherzustellen, dass alle Komponenten korrekt installiert sind.
    ```bash
    poetry run pytest
    ```
    Alle Tests sollten erfolgreich durchlaufen.

### b) Google Colab Setup

Für GPU-beschleunigtes Training ist Google Colab eine ausgezeichnete, kostenlose Option.

#### Schnellstart mit Colab Notebook (Empfohlen)

**Am einfachsten**: Verwenden Sie das fertige Colab Notebook `archimedes_colab.ipynb`:

1. **Öffne das Notebook in Colab**:
   - Laden Sie `archimedes_colab.ipynb` in Google Colab hoch, oder
   - Klonen Sie das Repository und öffnen Sie das Notebook

2. **Aktiviere GPU**:
   - `Laufzeit` → `Laufzeittyp ändern` → `Hardwarebeschleuniger: GPU`

3. **Führe die Zellen nacheinander aus**:
   - Das Notebook führt Sie durch Installation, Setup und Training

#### Manuelle Installation (Alternative)

Falls Sie das Notebook nicht verwenden möchten:

1.  **Öffne ein neues Colab Notebook**.

2.  **Stelle die Laufzeit auf GPU um**:
    `Laufzeit` → `Laufzeittyp ändern` → `Hardwarebeschleuniger: GPU`.

3.  **Klone das Repository**:
    ```python
    !git clone <repository_url>
    %cd Projekt-Archimedes
    ```

4.  **Installiere die Abhängigkeiten** (ohne Poetry):
    ```python
    # Installiere Basis-Dependencies
    !pip install -r requirements_colab.txt

    # Installiere PyTorch mit CUDA Support für Colab
    !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

5.  **Führe die Skripte aus**:
    ```python
    # Mit automatischer Konfiguration (empfohlen für Colab)
    !python train_end_to_end.py --auto-config --total-games 100

    # Oder mit manuellen Parametern (Colab-optimiert)
    !python train_end_to_end.py --num-workers 1 --total-games 100 --batch-size 32
    ```

#### Colab-spezifische Parameter-Empfehlungen

Das System erkennt automatisch Colab-Umgebungen und passt die Parameter an:

- **Workers**: Colab hat nur 2 CPU-Kerne → `--num-workers 1` (automatisch gesetzt)
- **Batch-Size**: 
  - T4 GPU: ~32 (automatisch)
  - A100 GPU: ~64 (automatisch)
- **Replay Buffer**: Reduziert auf ~20.000 für begrenzten RAM

**Tipp**: Verwenden Sie `--auto-config` für optimale Colab-Parameter!

#### Troubleshooting

**Problem**: "Out of Memory" Fehler
- **Lösung**: Reduzieren Sie `--batch-size` (z.B. `--batch-size 16`) oder `--replay-buffer-size`

**Problem**: Training läuft sehr langsam
- **Lösung**: Stellen Sie sicher, dass GPU aktiviert ist (`Laufzeit` → `Laufzeittyp ändern` → `GPU`)

**Problem**: Poetry-Installation schlägt fehl
- **Lösung**: Verwenden Sie `requirements_colab.txt` statt Poetry (siehe manuelle Installation oben)

## 5. Benutzung

### a) Datensatz erstellen (`create_dataset.py`)

Dieses Skript verarbeitet PGN-Dateien und bereitet sie für das Training vor.

**Argumente**:
*   `--input-dir`: (Pflicht) Verzeichnis, das deine `.pgn`-Dateien enthält.
*   `--output-dir`: (Pflicht) Verzeichnis, in das die Trainings-Shards (`.pt`-Dateien) gespeichert werden.
*   `--shard-size`: (Optional) Anzahl der Positionen pro Shard. Default: `10000`.
*   `--dataset-type`: (Pflicht) Art des zu erstellenden Datensatzes.
    *   `tpn`: Erstellt Daten für das TPN-Training: `(tensor_board, move_index, game_result)`.
    *   `san`: Erstellt Daten für das SAN-Training: `(graph_board, raw_comment_string)`.

**Beispiel**:
```bash
poetry run python src/archimedes/create_dataset.py \
    --input-dir ./data/pgn \
    --output-dir ./data/shards_tpn \
    --dataset-type tpn \
    --shard-size 50000
```

### b) System Benchmark (`benchmark_system.py`)

**NEU**: Dieses Skript benchmarkt deine Hardware (CPU, GPU, RAM) und schlägt optimale Trainingsparameter vor, um die volle Kapazität deines PCs auszunutzen, während der PC weiterhin nutzbar bleibt.

**Argumente**:
*   `--output`: (Optional) Ausgabedatei für Benchmark-Ergebnisse. Default: `benchmark_results.json`.
*   `--skip-gpu-test`: (Optional) Überspringt GPU-Tests für schnellere Ausführung.

**Beispiel**:
```bash
# Führe vollständigen Benchmark durch
poetry run python benchmark_system.py

# Benchmark ohne GPU-Tests (schneller)
poetry run python benchmark_system.py --skip-gpu-test
```

Das Skript testet:
- **CPU**: Anzahl Kerne, Geschwindigkeit, aktuelle Auslastung
- **RAM**: Gesamter/verfügbarer Speicher, Geschwindigkeit
- **GPU**: Speicher, Compute-Capability, optimale Batch-Size

**Automatische Parameter-Optimierung**:
- Reserviert automatisch CPU-Kerne für System-Nutzung (25% oder min. 2 Kerne)
- Findet optimale Batch-Size basierend auf GPU-Speicher
- Empfiehlt optimale Anzahl von Workers für DataLoader und Self-Play
- Berechnet optimale Replay-Buffer-Größe basierend auf verfügbarem RAM

### c) End-to-End-Training (`train_end_to_end.py`)

Dies ist das Hauptskript, um die KI durch Self-Play zu trainieren.

**Argumente**:
*   `--num-workers`: (Optional) Anzahl der parallelen Prozesse für die Generierung von Self-Play-Spielen. Auto-konfiguriert mit `--auto-config`.
*   `--total-games`: (Optional) Gesamtzahl der zu spielenden Partien für den Trainingslauf. Default: `10`.
*   `--batch-size`: (Optional) Batch-Size für Training. Auto-konfiguriert mit `--auto-config`.
*   `--device`: (Optional) Device (cuda/cpu). Auto-detektiert wenn nicht angegeben.
*   `--auto-config`: (Optional) **Verwendet Benchmark-Ergebnisse für optimale Konfiguration.**
*   `--benchmark-file`: (Optional) Pfad zur Benchmark-Datei. Default: `benchmark_results.json`.
*   `--replay-buffer-size`: (Optional) Größe des Replay-Buffers. Auto-konfiguriert mit `--auto-config`.

**Beispiele**:
```bash
# Training mit automatischer Konfiguration (empfohlen!)
poetry run python benchmark_system.py  # Erst Benchmark ausführen
poetry run python train_end_to_end.py --auto-config --total-games 1000

# Training mit manuellen Parametern
poetry run python train_end_to_end.py --num-workers 4 --total-games 1000 --batch-size 64
```
*Hinweis: Mit `--auto-config` werden alle Parameter automatisch basierend auf deiner Hardware optimiert, sodass du die volle Kapazität ausnutzt, während der PC weiterhin nutzbar bleibt.*

### d) TPN Training (`train_tpn.py`)

Trainiert das Tactical Perception Network isoliert.

**Argumente**:
*   `--dataset-dir`: (Pflicht) Verzeichnis mit Trainings-Shards.
*   `--epochs`: (Optional) Anzahl Epochen. Default: `10`.
*   `--batch-size`: (Optional) Batch-Size. Auto-konfiguriert mit `--auto-config`.
*   `--device`: (Optional) Device (cuda/cpu). Auto-detektiert wenn nicht angegeben.
*   `--auto-config`: (Optional) **Verwendet Benchmark-Ergebnisse für optimale Konfiguration.**
*   `--num-workers`: (Optional) DataLoader Workers. Auto-konfiguriert mit `--auto-config`.
*   `--pin-memory`: (Optional) Pin Memory für DataLoader. Auto-konfiguriert mit `--auto-config`.

**Beispiel**:
```bash
# Mit automatischer Konfiguration
poetry run python train_tpn.py --dataset-dir ./data/shards_tpn --auto-config --epochs 20
```

### e) SAN Training (`train_san.py`)

Trainiert das Strategic Abstraction Network isoliert.

**Argumente**:
*   `--dataset-dir`: (Pflicht) Verzeichnis mit Trainings-Shards.
*   `--epochs`: (Optional) Anzahl Epochen. Default: `10`.
*   `--batch-size`: (Optional) Batch-Size. Auto-konfiguriert mit `--auto-config`.
*   `--device`: (Optional) Device (cuda/cpu). Auto-detektiert wenn nicht angegeben.
*   `--auto-config`: (Optional) **Verwendet Benchmark-Ergebnisse für optimale Konfiguration.**
*   `--num-workers`: (Optional) DataLoader Workers. Auto-konfiguriert mit `--auto-config`.

**Beispiel**:
```bash
# Mit automatischer Konfiguration
poetry run python train_san.py --dataset-dir ./data/shards_san --auto-config --epochs 20
```

### f) Elo-Bewertung (`evaluate_elo.py`)

Dieses Skript spielt Partien zwischen zwei Modell-Versionen, um deren relative Spielstärke zu messen.

**Argumente**:
*   `--model1_path`: (Pflicht) Pfad zu den Gewichten des ersten Modells.
*   `--model2_path`: (Pflicht) Pfad zu den Gewichten des zweiten Modells.
*   `--num-games`: (Optional) Anzahl der zu spielenden Partien. Default: `10`.
*   `--egtb-path`: (Optional) Pfad zu den Endgame-Tablebases (Syzygy), um die Genauigkeit im Endspiel zu erhöhen.

**Beispiel**:
```bash
poetry run python evaluate_elo.py \
    --model1_path ./models/archimedes_v2.pth \
    --model2_path ./models/archimedes_v1.pth \
    --num-games 50
```

### g) TPN-Quantisierung (`quantize_tpn.py`)

Dieses Skript optimiert ein trainiertes TPN-Modell für eine schnellere Inferenz auf CPUs.

**Anwendung**:
```bash
# Lädt das TPN-Modell, quantisiert es und speichert es
# (Pfade müssen im Skript angepasst werden)
poetry run python quantize_tpn.py
```

---
**Projekt "Archimedes"** - Eine neue Ära der strategischen Schach-KI.
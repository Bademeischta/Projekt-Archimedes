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

1.  **Öffne ein neues Colab Notebook**.

2.  **Stelle die Laufzeit auf GPU um**:
    `Laufzeit` -> `Laufzeittyp ändern` -> `Hardwarebeschleuniger: GPU`.

3.  **Klone das Repository**:
    ```python
    !git clone <repository_url>
    %cd archimedes
    ```

4.  **Installiere die Abhängigkeiten**:
    ```python
    # Installiere Poetry
    !pip install poetry

    # Installiere Basis-Abhängigkeiten (ohne PyTorch)
    !poetry install

    # Installiere die passende PyTorch-Version für Colab (mit CUDA)
    # Colab hat in der Regel eine CUDA-Version vorinstalliert.
    # Überprüfe die Version mit !nvcc --version
    !poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118 # oder eine andere CUDA-Version
    ```

5.  **Führe die Skripte aus**:
    Du kannst die Python-Skripte direkt aus dem Notebook heraus ausführen:
    ```python
    !poetry run python train_end_to_end.py --num-workers 2 --total-games 50
    ```

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

### b) End-to-End-Training (`train_end_to_end.py`)

Dies ist das Hauptskript, um die KI durch Self-Play zu trainieren.

**Argumente**:
*   `--num-workers`: (Optional) Anzahl der parallelen Prozesse für die Generierung von Self-Play-Spielen. Default: `2`.
*   `--total-games`: (Optional) Gesamtzahl der zu spielenden Partien für den Trainingslauf. Default: `10`.

**Beispiel**:
```bash
# Starte das Training mit 4 Worker-Prozessen für insgesamt 1000 Partien
poetry run python train_end_to_end.py --num-workers 4 --total-games 1000
```
*Hinweis: Dieses Skript ist für verteiltes Training vorbereitet und kann mit `torch.distributed.run` für Multi-GPU- oder Multi-Node-Training gestartet werden.*

### c) Elo-Bewertung (`evaluate_elo.py`)

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

### d) TPN-Quantisierung (`quantize_tpn.py`)

Dieses Skript optimiert ein trainiertes TPN-Modell für eine schnellere Inferenz auf CPUs.

**Anwendung**:
```bash
# Lädt das TPN-Modell, quantisiert es und speichert es
# (Pfade müssen im Skript angepasst werden)
poetry run python quantize_tpn.py
```

---
**Projekt "Archimedes"** - Eine neue Ära der strategischen Schach-KI.
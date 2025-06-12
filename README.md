## Overview

This repository implements a three‐stage fraud detection pipeline combining a heterogeneous graph neural network (GNN) and an XGBoost classifier. The goal is to identify fraudulent transactions by:

1. **Constructing** a DGL heterogeneous graph over all transactions (train + validation), with three node types (`client`, `merchant`, `transaction`) and four directed edge types (`buys`, `bought`, `issued`, `sells`).
2. **Inferring** transaction embeddings via a pre‐trained GraphSAGE model on that graph.
3. **Classifying** each transaction embedding with an XGBoost model to produce a fraud score.

All code is orchestrated through `run.py`, and dependencies are managed via Conda in `environment.yml`. Sample data (`training.csv`, `validation.csv`) and pre‐trained model artifacts (`model/` folder) are provided.

---

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Directory Structure](#directory-structure)
* [Usage](#usage)
* [Pipeline Stages](#pipeline-stages)

  * [1. Graph Construction](#1-graph-construction)
  * [2. GraphSAGE Inference](#2-graphsage-inference)
  * [3. Classification](#3-classification)
* [Model Artifacts](#model-artifacts)
* [Visualization](#visualization)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* **Heterogeneous Graph**: Three node types (`client`, `merchant`, `transaction`) and four edge relations.
* **GraphSAGE Embeddings**: Pre‐trained GNN to capture transaction context.
* **XGBoost Classifier**: Uses learned embeddings to predict fraud probability.
* **Interactive Visualization**: Supports exporting the graph to an HTML file via PyVis.
* **Reproducible Environment**: `environment.yml` pins all dependencies, including PyTorch, DGL, cuDF, XGBoost, and PyVis.

---

## Prerequisites

* **OS**: Ubuntu/Linux x86\_64 (GPU with CUDA 11.8 or CPU‐only).
* **Tools**:

  * Git
  * Miniconda or Anaconda
  * (Optional) NVIDIA driver & CUDA 11.8 if using GPU acceleration

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/fraud-detection-pipeline.git
   cd fraud-detection-pipeline
   ```

2. **Create the Conda environment**

   ```bash
   conda env create --name gnn-fraud --file environment.yml
   ```

3. **Activate it**

   ```bash
   conda activate gnn-fraud
   ```

---

## Directory Structure

```
fraud-detection-pipeline/
├── run.py                  # Main entrypoint: orchestrates all stages
├── environment.yml         # Conda environment spec
├── training.csv            # Training data with 'fraud_label'
├── validation.csv          # Validation/inference data
│
├── stages/                 # Modular pipeline stages
│   ├── graph_construction_stage.py
│   ├── graph_sage_stage.py
│   └── classification_stage.py
│
└── model/                  # Pre-trained artifacts
    ├── graph.pkl           # Serialized DGL heterograph
    ├── hyperparams.pkl     # GNN hyperparameters
    ├── model.pt            # GraphSAGE weights
    └── xgb.pt              # XGBoost classifier
```

---

## Usage

Run the full pipeline with:

```bash
python run.py \
  --training_file training.csv \
  --input_file   validation.csv \
  --model_dir    model \
  --output_file  predictions.csv \
  --batch_size   128
```

Options:

* `--training_file`: CSV containing transactions labeled with `fraud_label`.
* `--input_file`: CSV of transactions to score (no `fraud_label`).
* `--model_dir`: Directory holding `graph.pkl`, `hyperparams.pkl`, `model.pt`, and `xgb.pt`.
* `--output_file`: Path to write the final `transaction_id,fraud_score` CSV.
* `--batch_size`: Batch size for GraphSAGE inference (default: 100).

---

## Pipeline Stages

### 1. Graph Construction

**File:** `stages/graph_construction_stage.py`

* Loads training + validation data via cuDF.
* Encodes `client_node`, `merchant_node`, and `index`.
* Builds a DGL heterograph with:

  * Node types: `client`, `transaction`, `merchant`
  * Edge types:

    * `client → transaction` (`buys`)
    * `transaction → client` (`bought`)
    * `transaction → merchant` (`issued`)
    * `merchant → transaction` (`sells`)
* Returns:

  * `graph`: `DGLHeteroGraph`
  * `node_features`: normalized `torch.Tensor`
  * `test_index`: indices of validation transactions

### 2. GraphSAGE Inference

**File:** `stages/graph_sage_stage.py`

* Loads the GraphSAGE model and parameters with `load_model()`.
* Runs inductive inference on `test_index` nodes:

  ```python
  embeddings, _ = model.inference(graph, node_features, test_index)
  ```
* Appends embedding columns (`ind_emb_0`…`ind_emb_k`) to the inference cuDF DataFrame.

### 3. Classification

**File:** `stages/classification_stage.py`

* Reloads embeddings and optionally an XGBoost model (`xgb.pt`).
* If XGBoost present, predicts:

  ```python
  scores = xgb_model.predict_proba(embeddings)[:, 1]
  ```

  Otherwise, uses GNN softmax on logits.
* Writes `predictions.csv` with:

  ```
  transaction_id,fraud_score
  ```

---

## Model Artifacts

* **graph.pkl**: Pickled DGL heterograph.
* **hyperparams.pkl**: Dictionary with:

  ```json
  {
    "in_size": int,
    "hidden_size": int,
    "out_size": int,
    "n_layers": int,
    "embedding_size": int,
    "target_node": "transaction"
  }
  ```
* **model.pt**: PyTorch state dict for GraphSAGE.
* **xgb.pt**: Serialized XGBoost model.

---

## Visualization

Use the built-in `visualize_graph()` (or your own module) to export an interactive HTML:

```python
from stages.graph_construction_stage import construct_fraud_graph
from your_viz_module import visualize_graph
import cudf

df_train = cudf.read_csv("training.csv")
visualize_graph(
  train_data=df_train,
  col_drop=["client_node","merchant_node","index"],
  partition="validation",
  html_path="hetero_graph_validation.html"
)
```

Open `hetero_graph_validation.html` in a browser to pan, zoom, and inspect node/edge types.

---
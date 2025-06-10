import os
import pickle

import cudf                                           # GPU DataFrame I/O :contentReference[oaicite:2]{index=2}
import torch
from .model import load_model
from .graph_construction_stage import construct_fraud_graph

import cuml


class ClassificationStage:
    """
    Loads a trained GNN (+ optional XGBoost) and produces fraud scores.
    """

    def __init__(self,
                 model_dir: str,
                 batch_size: int = 100):
        # Load GNN and its hyperparameters
        self.model, self.graph, self.hyperparams = load_model(model_dir)
        self.batch_size = batch_size
        self.model_dir = model_dir

    def run(self,
            train_csv: str,
            infer_csv: str,
            output_csv: str):
        """
        Parameters
        ----------
        train_csv : str
            Path to training CSV (used to reconstruct the graph).
        infer_csv : str
            Path to inference CSV (contains transactions to score).
        output_csv : str
            Path where fraud scores CSV will be written.
        """
        # 1. Build graph, features, and test indices
        graph, features, test_idx = construct_fraud_graph(train_csv,
                                                          infer_csv)

        # 2. GNN inference: get embeddings or softmax scores
        self.model.eval()
        with torch.no_grad():
            # embeddings for test nodes
            embeddings, _ = self.model.inference(
                graph,
                features,
                test_idx,
                target_node=self.hyperparams.get("target_node", "transaction"),
                batch_size=self.batch_size
            )

        # 3. Optional XGBoost classifier on embeddings
        xgb_path = os.path.join(self.model_dir, "xgb.pt")
        if os.path.exists(xgb_path):
            # with open(xgb_path, "rb") as f:
            #     xgb_model = pickle.load(f)
            xgb_model = cuml.ForestInference.load(xgb_path, output_class=True, model_type="xgboost")  # cuml ForestInference for XGBoost
            # assumes scikit-learn API
            scores = xgb_model.predict_proba(embeddings)[:, 1]
        else:
            # fallback: use GNN softmax on logits
            with torch.no_grad():
                logits, _ = self.model(self.graph, features)
            scores = torch.softmax(logits, dim=1)[test_idx, 1].cpu().numpy()  # Softmax API :contentReference[oaicite:3]{index=3}

        # 4. Assemble output DataFrame and write CSV
        df_infer = cudf.read_csv(infer_csv)
        df_out = cudf.DataFrame({
            "transaction_id": df_infer["transaction_id"].iloc[test_idx].to_pandas().values,
            "fraud_score":    scores
        })
        df_out.to_csv(output_csv, index=False)

import os
import pickle

import cudf
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay

from .model import load_model
from .graph_construction_stage import construct_fraud_graph

import cuml  # for XGBoost inference, if available


class ClassificationStage:
    """
    Loads a trained GNN (+ optional XGBoost) and produces fraud scores,
    with confusion matrix and ROC curve evaluations.
    """

    def __init__(self,
                 model_dir: str,
                 batch_size: int = 100):
        self.model, self.graph, self.hyperparams = load_model(model_dir)
        self.batch_size = batch_size
        self.model_dir = model_dir

    def run(self,
            train_csv: str,
            infer_csv: str,
            output_csv: str,
            fig_dir: str = "./figures"):
        # Ensure output directory exists
        os.makedirs(fig_dir, exist_ok=True)

        # Step 1: Build graph, features, and train/test indices
        graph, features, train_idx, test_idx = construct_fraud_graph(train_csv, infer_csv)

        # Extract ground truth labels
        df_train = cudf.read_csv(train_csv)
        df_infer = cudf.read_csv(infer_csv)
        y_true_train = df_train["fraud_label"].iloc[train_idx.to_pandas()].to_numpy()
        y_true_test  = df_infer["fraud_label"].iloc[test_idx.to_pandas()].to_numpy()

        # Step 2: GNN inference
        self.model.eval()
        with torch.no_grad():
            embeddings, _ = self.model.inference(
                graph,
                features,
                test_idx,
                target_node=self.hyperparams.get("target_node", "transaction"),
                batch_size=self.batch_size
            )

        # Step 3: Optional XGBoost classifier
        xgb_path = os.path.join(self.model_dir, "xgb.pt")
        if os.path.exists(xgb_path):
            xgb_model = cuml.ForestInference.load(xgb_path, output_class=True, model_type="xgboost")
            y_score_test = xgb_model.predict_proba(embeddings)[:, 1]
        else:
            logits, _ = self.model(self.graph, features)
            y_probs = torch.softmax(logits, dim=1)
            y_score_test = y_probs[test_idx, 1].cpu().numpy()

        # Predicted labels using a 0.5 threshold
        y_pred_test = (y_score_test >= 0.5).astype(int)

        # Step 4: Save output CSV
        df_out = cudf.DataFrame({
            "transaction_id": df_infer["index"].iloc[test_idx].to_pandas().values,
            "fraud_score":    y_score_test
        })
        df_out.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")

        # Step 5: Confusion Matrix (Test data)
        cm = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format="d")
        ax_cm.set_title("Test Confusion Matrix")
        fig_cm.tight_layout()
        cm_path = os.path.join(fig_dir, "confusion_matrix.png")
        fig_cm.savefig(cm_path)
        plt.close(fig_cm)
        print(f"Confusion matrix saved to {cm_path}")

        # Step 6: ROC Curve (Test data)
        fpr, tpr, _ = roc_curve(y_true_test, y_score_test, pos_label=1)
        auc_score = roc_auc_score(y_true_test, y_score_test)
        disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score, name="Test ROC")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        disp_roc.plot(ax=ax_roc)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_title(f"ROC Curve (AUC = {auc_score:.3f})")
        fig_roc.tight_layout()
        roc_path = os.path.join(fig_dir, "roc_curve.png")
        fig_roc.savefig(roc_path)
        plt.close(fig_roc)
        print(f"ROC curve saved to {roc_path}")
import cudf
import torch

from .model import load_model, build_fsi_graph, prepare_data, train_hinsage


class GraphSAGEStage:
    """
    Performs inductive inference with a pre‐trained DGL GraphSAGE model.

    Usage:
        stage = GraphSAGEStage(model_dir="model/")
        df_out = stage.run(
            df_input,        # cudf.DataFrame with your records
            graph,           # DGLHeteroGraph built previously
            node_features,   # torch.Tensor of all node features
            test_index       # torch.LongTensor of nodes to infer
        )
    """

    def __init__(self,
                 model_dir: str,
                 batch_size: int = 100,
                 record_id: str = "index"):
        # loads graph, hyperparams, and model weights
        self.model, self.graph, self.hyperparams = load_model(model_dir)
        self.batch_size = batch_size
        self.record_id = record_id

    def run(self,
            df_input: cudf.DataFrame,
            graph: torch.nn.Module,
            node_features: torch.Tensor,
            test_index: torch.LongTensor
            ) -> cudf.DataFrame:
        """
        Runs GraphSAGE inference and appends embeddings to df_input.

        Parameters
        ----------
        df_input : cudf.DataFrame
            Original inference DataFrame containing at least self.record_id column.
        graph : dgl.DGLHeteroGraph
            The heterogeneous graph on which to run inference.
        node_features : torch.Tensor
            Float tensor of shape (N, F) with node features.
        test_index : torch.LongTensor
            1D tensor of length M giving the node indices in df_input to embed.

        Returns
        -------
        cudf.DataFrame
            A copy of df_input with new columns "ind_emb_0", "ind_emb_1", … appended.
        """

        # 1) Prepare data
        train_df = cudf.read_csv("training.csv")
        val_df   = cudf.read_csv("validation.csv")
        td, vd, tidx, vidx, labels, all_df = prepare_data(train_df, val_df)
        g, feats = build_fsi_graph(all_df, ["client_node","merchant_node","index"])

        # 2) Train
        model = train_hinsage(
            graph=g,
            features=feats,
            train_idx=tidx,
            labels=labels,
            in_size=feats.shape[1],
            save_path="model"
        )

        # 1. perform inductive inference
        # embeddings, _ = self.model.inference(
        #     graph,
        #     node_features,
        #     test_index,
        #     target_node=self.hyperparams.get("target_node", "transaction"),
        #     batch_size=self.batch_size
        # )

        embeddings, _ = model.inference(
            graph,
            node_features,
            test_index,
            target_node=self.hyperparams.get("target_node", "transaction"),
            batch_size=self.batch_size
        )

        # 2. convert to cudf DataFrame
        df_emb = cudf.DataFrame(embeddings.cpu().numpy())

        # 3. rename columns
        df_emb = df_emb.rename(
            columns={i: f"ind_emb_{i}" for i in df_emb.columns},
            copy=False
        )

        # 4. merge embeddings into original DataFrame
        df_out = df_input.reset_index(drop=True).copy()
        for col in df_emb.columns:
            df_out[col] = df_emb[col]

        return df_out

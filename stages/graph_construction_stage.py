import cudf
import torch

from .model import prepare_data, build_fsi_graph, visualize_graph

def construct_fraud_graph(
    training_csv: str,
    inference_csv: str
):
    """
    Reads the training and inference CSVs (as cuDF DataFrames),
    prepares combined data, and builds the DGL hetero-graph.

    Returns:
      graph         : DGLHeteroGraph
      node_features : torch.Tensor (float32)
      test_index    : torch.LongTensor
    """
    # Load CSVs with cuDF
    train_df = cudf.read_csv(training_csv)
    infer_df = cudf.read_csv(inference_csv)

    # Prepare combined data: split, encode indices, extract labels
    train_part, infer_part, train_idx, infer_idx, labels, combined = \
        prepare_data(train_df, infer_df)

    # Build DGL graph and feature tensor
    graph, node_features = build_fsi_graph(
        combined,
        col_drop=['client_node', 'merchant_node', 'index']
    )

    # Visualize the graph structure
    visualize_graph(
        train_df,
        col_drop=['client_node', 'merchant_node', 'index'],
        partition="train"
    )

    visualize_graph(
        infer_df,
        col_drop=['client_node', 'merchant_node', 'index'],
        partition="infer"
    )

    visualize_graph(
        cudf.concat([train_df, infer_df], axis=0),
        col_drop=['client_node', 'merchant_node', 'index'],
        partition="combined"
    )

    # Convert test indices from cuDF to torch.LongTensor via DLPack
    test_index = torch.from_dlpack(infer_idx.values.toDlpack()).long()

    return graph, node_features.float(), test_index
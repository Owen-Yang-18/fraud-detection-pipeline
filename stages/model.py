import os
import pickle
import typing

import cupy
import dgl
import torch
from dgl import nn as dglnn
from torch import nn
from torch.nn import functional as F
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

import cudf as cf
import pandas as pd


class BaseHeteroGraph(nn.Module):
    """
    Base class for Heterogeneous Graph Neural Network (GNN) models.

    Parameters
    ----------
    input_graph : dgl.DGLHeteroGraph
        The input graph on which the HeteroRGCN operates. It should be a heterogeneous graph.
    embedding_size : int
        The size of the node embeddings learned during training.
    target : str, optional
        The target attribute for which the node representations are learned.
    """

    def __init__(self, input_graph: dgl.DGLHeteroGraph, embedding_size: int, target: str):

        super().__init__()
        self._target = target

        # categorical embeding
        self.hetro_embedding = dglnn.HeteroEmbedding(
            {ntype: input_graph.number_of_nodes(ntype)
             for ntype in input_graph.ntypes if ntype != self._target},
            embedding_size)

        self.layers = nn.ModuleList()

    def forward(self, input_graph: dgl.DGLHeteroGraph, features: torch.tensor) -> (torch.tensor, torch.tensor):

        # Get embeddings for all none target node types.
        h_dict = self.hetro_embedding(
            {ntype: input_graph[0].nodes(ntype)
             for ntype in self.hetro_embedding.embeds.keys()})

        h_dict[self._target] = features

        # Forward pass to layers.
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(input_graph[i], h_dict)

        embedding = h_dict[self._target]

        return self.layers[-1](embedding), embedding

    def infer(self, input_graph: dgl.DGLHeteroGraph, features: torch.tensor) -> (torch.tensor, torch.tensor):
        """Perform inference through forward pass

        Parameters
        ----------
        input_graph : dgl.DGLHeteroGraph
            input inference graph
        features : torch.tensor
            node features

        Returns
        -------
        torch.tensor, torch.tensor
            prediction, feature embedding
        """

        predictions, embedding = self(input_graph, features)
        return nn.Sigmoid()(predictions), embedding

    @torch.no_grad()
    def evaluate(self, eval_loader: dgl.dataloading.DataLoader, feature_tensors: torch.Tensor,
                 target_node: str) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Evaluate the specified model on the given evaluation input graph

        Parameters
        ----------
        eval_loader : dgl.dataloading.DataLoader
            DataLoader containing the evaluation dataset.
        feature_tensors : torch.Tensor
            The feature tensors corresponding to the evaluation data.
            Shape: (num_samples, num_features).
        target_node : str
            The target node for evaluation, indicating the node of interest.

        Returns
        -------
        (torch.Tensor, torch.Tensor, torch.Tensor)
            A tuple containing numpy arrays of logits, eval seed and embeddings
        """

        self.eval()
        eval_logits = []
        eval_seeds = []
        embedding = []

        for _, output_nodes, blocks in eval_loader:

            seed = output_nodes[target_node]

            nid = blocks[0].srcnodes[target_node].data[dgl.NID]
            input_features = feature_tensors[nid]
            logits, embedd = self.infer(blocks, input_features)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seed)
            embedding.append(embedd)

        eval_logits = torch.cat(eval_logits)
        eval_seeds = torch.cat(eval_seeds)
        embedding = torch.cat(embedding)
        return eval_logits, eval_seeds, embedding

    def inference(self,
                  input_graph: dgl.DGLHeteroGraph,
                  feature_tensors: torch.Tensor,
                  test_idx: torch.Tensor,
                  target_node: str = "transaction",
                  batch_size: int = 100) -> (torch.Tensor, torch.Tensor):
        """
        Perform inference on a given model using the provided input graph and feature tensors.

        Parameters
        ----------
        input_graph : dgl.DGLHeteroGraph
            The input heterogeneous graph in DGL format. It represents the graph structure.
        feature_tensors : torch.Tensor
            The input feature tensors for nodes in the input graph. Each row corresponds to the features of a single
            node.
        test_idx : torch.Tensor
            The indices of the nodes in the input graph that are used for testing and evaluation.
        target_node : str, optional (default: "transaction")
            The type of node for which inference will be performed. By default, it is set to "transaction".
        batch_size : int, optional (default: 100)
            The batch size used during inference to process data in mini-batches.

        Returns
        -------
        test_embedding : torch.Tensor
            The embedding of for the target nodes obtained from the model's inference.
        test_seed: torch.Tensor
            The seed of the target nodes used for inference.
        """

        # create sampler and test dataloaders
        full_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=3)
        test_dataloader = dgl.dataloading.DataLoader(input_graph, {target_node: test_idx},
                                                     full_sampler,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=0)
        _, test_seed, test_embedding = self.evaluate(test_dataloader, feature_tensors, target_node)

        return test_embedding, test_seed


class HeteroRGCN(BaseHeteroGraph):
    """
    Heterogeneous Relational Graph Convolutional Network (HeteroRGCN) model.

    This class represents a Heterogeneous Relational Graph Convolutional Network (HeteroRGCN)
    used for node representation learning in heterogeneous graphs.

    Parameters
    ----------
    input_graph : dgl.DGLHeteroGraph
        The input graph on which the HeteroRGCN operates. It should be a heterogeneous graph.
    in_size : int
        The input feature size for each node in the graph.
    hidden_size : int
        The number of hidden units or feature size of the nodes after the convolutional layers.
    out_size : int
        The output feature size for each node. This will be the final representation size of each node.
    n_layers : int
        The number of graph convolutional layers in the HeteroRGCN model.
    embedding_size : int
        The size of the node embeddings learned during training.
    target : str, optional
        The target attribute for which the node representations are learned.
        Default is 'transaction'.

    Attributes
    ----------
    input_graph : dgl.DGLHeteroGraph
        The input graph on which the HeteroRGCN operates.
    in_size : int
        The input feature size for each node in the graph.
    hidden_size : int
        The number of hidden units or feature size of the nodes after the convolutional layers.
    out_size : int
        The output feature size for each node. This will be the final representation size of each node.
    n_layers : int
        The number of graph convolutional layers in the HeteroRGCN model.
    embedding_size : int
        The size of the node embeddings learned during training
    target : str
        The target attribute for which the node representations are learned.

    Methods
    -------
    forward(input_graph, features)
        Forward pass of the HeteroRGCN model.

    Notes
    -----
    HeteroRGCN is a deep learning model designed for heterogeneous graphs.
    It applies graph convolutional layers to learn representations of nodes in the graph.

    Examples
    --------
    >>> input_graph = dgl.heterograph(...)
    >>> in_size = 16
    >>> hidden_size = 32
    >>> out_size = 64
    >>> n_layers = 2
    >>> embedding_size = 128
    >>> target = 'transaction'
    >>> model = HeteroRGCN(input_graph, in_size, hidden_size, out_size, n_layers, embedding_size, target='transaction')
    """

    def __init__(self,
                 input_graph: dgl.DGLHeteroGraph,
                 in_size: int,
                 hidden_size: int,
                 out_size: int,
                 n_layers: int,
                 embedding_size: int,
                 target: str = 'transaction'):

        super().__init__(input_graph=input_graph, embedding_size=embedding_size, target=target)

        # input size
        in_sizes = {
            rel: in_size if src_type == self._target else embedding_size
            for src_type,
            rel,
            _ in input_graph.canonical_etypes
        }

        self.layers.append(
            dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_sizes[rel], hidden_size)
                                   for rel in input_graph.etypes},
                                  aggregate='sum'))

        for _ in range(n_layers - 1):
            self.layers.append(
                dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hidden_size, hidden_size)
                                       for rel in input_graph.etypes},
                                      aggregate='sum'))

        # output layer
        self.layers.append(nn.Linear(hidden_size, out_size))


class HinSAGE(BaseHeteroGraph):
    """
    Heterogeneous GraphSAGE (HinSAGE) module for graph-based semi-supervised learning.

    This module implements a variant of GraphSAGE (Graph Sample and Aggregated) for heterogeneous graphs,
    where different types of nodes have distinct feature spaces.

    Parameters
    ----------
    input_graph : dgl.DGLHeteroGraph
        The input heterogeneous graph on which the HinSAGE model will operate.
    in_size : int
        The input feature size for each node type. This represents the dimensionality of the node features.
    hidden_size : int
        The size of the hidden layer(s) in the GraphSAGE model.
    out_size : int
        The output size for each node type. This represents the dimensionality of the output node embeddings.
    n_layers : int
        The number of GraphSAGE layers in the HinSAGE model.
    embedding_size : int
        The size of the final node embeddings after aggregation. This will be used as input for the downstream task.
    target : str, optional
        The target node type for the downstream task. Default is 'transaction'.
    aggregator_type : str, optional
        The type of aggregator to use for aggregation. Default is 'mean'.

    Methods
    -------
    forward(input_graph, features)
        Forward pass of the HinSAGE model.

    Notes
    -----
    HinSAGE is designed for semi-supervised learning on heterogeneous graphs. The model generates node embeddings by
    sampling and aggregating neighbor information from different node types in the graph.

    The target parameter specifies the node type for the downstream task. The model will only return embeddings for
    nodes of the specified type.

    Examples
    --------
    >>> input_graph = dgl.heterograph(...)  # Replace ... with actual heterogeneous graph data
    >>> in_size = 64
    >>> hidden_size = 128
    >>> out_size = 64
    >>> n_layers = 2
    >>> embedding_size = 256
    >>> target_node_type = 'transaction'
    >>> hinsage_model = HinSAGE(input_graph, in_size, hidden_size, out_size, n_layers, embedding_size, target_node_type)
    """

    def __init__(self,
                 input_graph: dgl.DGLHeteroGraph,
                 in_size: int,
                 hidden_size: int,
                 out_size: int,
                 n_layers: int,
                 embedding_size: int,
                 target: str = 'transaction',
                 aggregator_type: str = 'mean'):

        super().__init__(input_graph=input_graph, embedding_size=embedding_size, target=target)

        # create input features
        in_feats = {rel: embedding_size if rel != self._target else in_size for rel in input_graph.ntypes}

        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    rel:
                        dglnn.SAGEConv(
                            (in_feats[src_type], in_feats[v_type]), hidden_size, aggregator_type=aggregator_type)
                    for src_type,
                    rel,
                    v_type in input_graph.canonical_etypes
                },
                aggregate='sum'))

        for _ in range(n_layers - 1):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        rel: dglnn.SAGEConv(
                            hidden_size,
                            hidden_size,
                            aggregator_type=aggregator_type,
                        )
                        for rel in input_graph.etypes
                    },
                    aggregate='sum'))

        # output layer
        self.layers.append(nn.Linear(hidden_size, out_size))


# ──────────────────────────────────────────────────────────────────────────────
#  HINSAGE TRAINING HELPER
# ──────────────────────────────────────────────────────────────────────────────

def train_hinsage(
    graph: dgl.DGLHeteroGraph,
    features: torch.Tensor,
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    in_size: int,
    hidden_size: int = 64,
    out_size: int = 2,
    n_layers: int = 2,
    embedding_size: int = 1,
    fanouts: list[int] = [3, 3],
    batch_size: int = 100,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: typing.Optional[torch.device] = None,
    save_path: str = "model"
) -> HinSAGE:
    """
    Train a HinSAGE model end-to-end on the given heterogeneous graph.

    Parameters
    ----------
    graph : dgl.DGLHeteroGraph
        Full graph including train & validation nodes.
    features : torch.Tensor
        Node features tensor of shape (N, F).
    train_idx : torch.Tensor
        1D tensor of transaction node indices to train on.
    labels : torch.Tensor
        1D tensor of shape (N,) with integer labels (0/1) for fraud.
    in_size : int
        Number of input features for 'transaction' nodes.
    hidden_size, out_size, n_layers, embedding_size : int
        Model hyperparameters.
    fanouts : list[int]
        Number of neighbors to sample at each layer.
    batch_size : int
        Mini-batch size for neighbor sampling.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    device : torch.device, optional
        CUDA or CPU. If None, auto-selects.
    save_path : str
        Directory to save `hinsage.pt` after training.

    Returns
    -------
    Hinsage model (trained)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = graph.to(device)
    features = features.to(device)
    train_idx = train_idx.to(device)
    labels = labels.to(device)

    # 1. DataLoader with neighbor sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        graph,
        {'transaction': train_idx},
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )

    # 2. Model, loss, optimizer
    model = HinSAGE(
        graph,
        in_size=in_size,
        hidden_size=hidden_size,
        out_size=out_size,
        n_layers=n_layers,
        embedding_size=embedding_size,
        target="transaction"
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for input_nodes, output_nodes, blocks in dataloader:
            # a) features for this batch
            h = features[input_nodes['transaction']]
            # b) forward
            logits, _ = model(blocks, h)
            batch_labels = labels[output_nodes['transaction']]
            # c) backward
            loss = loss_fn(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg:.4f}")

    # 4. Save weights
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "hinsage.pt"))
    print(f"Trained HinSAGE weights saved to {save_path}/hinsage.pt")

    return model

def load_model(model_dir: str,
               gnn_model: BaseHeteroGraph = HinSAGE,
               device: torch.device = None) -> (BaseHeteroGraph, dgl.DGLHeteroGraph, dict):
    """Load trained models from model directory

    Parameters
    ----------
    model_dir : str
        models directory path
    gnn_model: BaseHeteroGraph
        GNN model type either HeteroRGCN or HinSAGE. Default HinSAGE
    device : torch.device, optional
        The device where the model and tensors should be loaded,
        by default torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Returns
    -------
    (nn.Module, dgl.DGLHeteroGraph, dict)
        model, training graph, hyperparameter
    """

    with open(os.path.join(model_dir, "graph.pkl"), 'rb') as f:
        graph = pickle.load(f)
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'rb') as f:
        hyperparameters = pickle.load(f)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = gnn_model(graph,
                      in_size=hyperparameters['in_size'],
                      hidden_size=hyperparameters['hidden_size'],
                      out_size=hyperparameters['out_size'],
                      n_layers=hyperparameters['n_layers'],
                      embedding_size=hyperparameters['embedding_size'],
                      target=hyperparameters['target_node']).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), weights_only=True))

    return model, graph, hyperparameters

def build_fsi_graph(train_data: cf.DataFrame, col_drop: list[str]) -> (dgl.DGLHeteroGraph, torch.Tensor):
    """Build a heterogeneous graph from an edgelist and node index.
    Parameters
    ----------
    train_data : cudf.DataFrame
        Training data containing node features.
    col_drop : list
        List of features to drop from the node features.
    Returns
    -------
    tuple
        A tuple containing the following elements:
        dgl.DGLHeteroGraph
            The built DGL graph representing the heterogeneous graph.
        torch.tensor
            Normalized feature tensor after dropping specified columns.
    Notes
    -----
    This function takes the training data, represented as a cudf DataFrame,
    and constructs a heterogeneous graph (DGLGraph) from the given edgelist
    and node index.

    The `col_drop` list specifies which features should be dropped from the
    node features to build the normalized feature tensor.

    Example
    -------
    >>> import cudf
    >>> train_data = cudf.DataFrame({'node_id': [1, 2, 3],
    ...                            'feature1': [0.1, 0.2, 0.3],
    ...                            'feature2': [0.4, 0.5, 0.6]})
    >>> col_drop = ['feature2']
    >>> graph, features = build_heterograph(train_data, col_drop)
    """

    feature_tensors = train_data.drop(col_drop, axis=1).values
    feature_tensors = torch.from_dlpack(feature_tensors.toDlpack())
    feature_tensors = (feature_tensors - feature_tensors.mean(0, keepdim=True)) / (0.0001 +
                                                                                   feature_tensors.std(0, keepdim=True))
    # Create client, merchant, transaction node id tensors & move to torch.tensor
    # col_drop column expected to be in ['client','merchant', 'transaction'] order to match
    # torch.tensor_split order
    client_tensor, merchant_tensor, transaction_tensor = torch.tensor_split(
        torch.from_dlpack(train_data[col_drop].values.toDlpack()).long(), 3, dim=1)

    client_tensor, merchant_tensor, transaction_tensor = (client_tensor.view(-1),
                                                          merchant_tensor.view(-1),
                                                          transaction_tensor.view(-1))

    edge_list = {
        ('client', 'buy', 'transaction'): (client_tensor, transaction_tensor),
        ('transaction', 'bought', 'client'): (transaction_tensor, client_tensor),
        ('transaction', 'issued', 'merchant'): (transaction_tensor, merchant_tensor),
        ('merchant', 'sell', 'transaction'): (merchant_tensor, transaction_tensor)
    }
    graph = dgl.heterograph(edge_list)

    return graph, feature_tensors

import cudf
from pyvis.network import Network

def visualize_graph_v2(
    df: cudf.DataFrame,
    client_col: str,
    merchant_col: str,
    txn_col: str,
    fraud_label_col: str,
    partition: str,
    html_path: str = None
):
    """
    Build an interactive PyVis network directly from a cuDF DataFrame,
    without DGL or NetworkX. Nodes and edges are added based on the
    client, merchant, and transaction columns. Transaction nodes are
    colored by fraud_label (0 vs. 1) using two shades of pink; other
    nodes use fixed skyblue/orange. Edges are colored by relation.
    """
    # 1. Default html path
    if html_path is None:
        html_path = f"heterograph-{partition}.html"

    # 2. Color mappings
    tx_color_map = {0: "#ffe6f0", 1: "#ff66b2"}  # light vs medium pink
    node_color_map = {
        "client":   "skyblue",
        "merchant": "orange"
    }
    edge_color_map = {
        "buy":     "red",
        "bought":  "blue",
        "issued":  "green",
        "sell":    "purple"
    }

    # 3. Instantiate PyVis network
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.toggle_physics(True)

    # 4. Add unique client nodes
    for cid in df[client_col].unique().to_pandas().tolist():
        net.add_node(
            cid,
            label=str(cid),
            color=node_color_map["client"],
            title=f"client {cid}"
        )

    # 5. Add unique merchant nodes
    for mid in df[merchant_col].unique().to_pandas().tolist():
        net.add_node(
            mid,
            label=str(mid),
            color=node_color_map["merchant"],
            title=f"merchant {mid}"
        )

    # 6. Add unique transaction nodes, colored by fraud_label
    #    We assume fraud_label_col aligns 1:1 with txn_col in df
    tx_df = df[[txn_col, fraud_label_col]].drop_duplicates().to_pandas()
    for _, row in tx_df.iterrows():
        tid = row[txn_col]
        lbl = int(row[fraud_label_col])
        net.add_node(
            tid,
            label=str(tid),
            color=tx_color_map[lbl],
            title=f"transaction {tid} — {'FRAUD' if lbl == 1 else 'OK'}"
        )

    # 7. Add edges for each relation per row
    for idx, row in df.to_pandas().iterrows():
        c = row[client_col]
        m = row[merchant_col]
        t = row[txn_col]

        # client → transaction ("buy")
        net.add_edge(c, t, color=edge_color_map["buy"],   title="buy")
        # transaction → client ("bought")
        net.add_edge(t, c, color=edge_color_map["bought"], title="bought")
        # transaction → merchant ("issued")
        net.add_edge(t, m, color=edge_color_map["issued"], title="issued")
        # merchant → transaction ("sell")
        net.add_edge(m, t, color=edge_color_map["sell"],   title="sell")

    # 8. Save the interactive HTML
    net.save_graph(html_path)
    print(f"Interactive graph (v2) saved to {html_path}")


def visualize_graph(
    train_data: cf.DataFrame,
    col_drop: list[str],
    partition: str,
):
    """
    Build a DGL heterograph, then create an interactive directed
    PyVis visualization with per–node‐type and per–edge‐type colors.
    """
    # 1. Extract fraud labels
    dlpack = train_data["fraud_label"].values.toDlpack()
    tx_labels = torch.from_dlpack(dlpack).long()

    # Define pink shades for transaction nodes
    # 0 → light pink; 1 → darker pink
    tx_color_map = {
        0: "#ffe6f0",  # very light pink
        1: "#ff66b2"   # medium pink
    }

    # Colors for other node types
    node_color_map = {
        "client":   "skyblue",
        "merchant": "orange"
    }

    # Colors for edges
    edge_color_map = {
        "buy":     "red",
        "bought":  "blue",
        "issued":  "green",
        "sell":    "purple"
    }

    # --- build IDs and graph on GPU ---
    id_arr = torch.from_dlpack(train_data[col_drop].values.toDlpack()).long()
    c_ids, m_ids, t_ids = torch.tensor_split(id_arr, 3, dim=1)
    c_ids, m_ids, t_ids = [x.view(-1) for x in (c_ids, m_ids, t_ids)]

    g = dgl.heterograph({
        ('client',      'buy',    'transaction'): (c_ids, t_ids),
        ('transaction', 'bought',  'client'):      (t_ids, c_ids),
        ('transaction', 'issued',  'merchant'):    (t_ids, m_ids),
        ('merchant',    'sell',   'transaction'): (m_ids, t_ids)
    })

    # 3. Save original indices per transaction node
    #    This ensures g.nodes['transaction'].data['orig_idx'][i] == train_data row index
    num_tx = g.num_nodes('transaction')
    g.nodes['transaction'].data['orig_idx'] = torch.arange(num_tx)  # 0…num_tx-1

    # 4. Convert to NetworkX, including orig_idx and node types
    g_cpu = g.to('cpu')
    nx_g = dgl.to_networkx(
        g_cpu,
        node_attrs=['ntype', 'orig_idx'],    # carry over node type and original index
        edge_attrs=['etype'],                # carry over relation name
        to_directed=True
    )

    # # --- move to CPU and convert to a directed NetworkX graph ---
    # g_cpu = g.to('cpu')
    # # to_directed=True converts hetero‐edges into a DiGraph
    # nx_g = dgl.to_networkx(
    #     g_cpu
    # )

    # --- build PyVis Network ---
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="black",
        notebook=False
    )
    net.toggle_physics(True)

    # --- add nodes with color by type ---
    # for node, data in nx_g.nodes(data=True):
    #     ntype = data.get('ntype', '')  # e.g. 'client'
    #     net.add_node(
    #         node,
    #         label=str(node),
    #         color=node_color_map.get(ntype, 'gray'),
    #         title=f"{ntype} {node}"
    #     )
    
    # 6. Add nodes with appropriate colors and annotations
    for node_id, data in nx_g.nodes(data=True):
        ntype = data.get("ntype", "")
        if ntype == "transaction":
            label = int(tx_labels[node_id].item())
            color = tx_color_map[label]
            title = f"transaction {node_id} — {'FRAUD' if label == 1 else 'OK'}"
        else:
            color = node_color_map.get(ntype, "gray")
            title = f"{ntype} {node_id}"

        net.add_node(
            node_id,
            label=str(node_id),
            color=color,
            title=title
        )

    # --- add edges with color by relation ---
    for src, dst, data in nx_g.edges(data=True):
        rel = data['etype'][1]  # the middle element is the relation name
        net.add_edge(
            src,
            dst,
            color=edge_color_map.get(rel, 'black'),
            title=rel
        )

    # --- generate and save HTML ---
    net.save_graph(f"heterograph-{partition}.html")


def prepare_data(
    training_data: cf.DataFrame, test_data: cf.DataFrame
) -> typing.Union[cf.DataFrame, cf.DataFrame, cf.Series, cf.Series, cupy.ndarray, cf.DataFrame]:
    """Process data for training/inference operation

    Parameters
    ----------
    training_data : cudf.DataFrame
        training data
    test_data : cudf.DataFrame
        test/validation data

    Returns
    -------
    tuple
     tuple of (training_data, test_data, train_index, test_index, label, combined data)
    """

    train_size = training_data.shape[0]
    cdf = cf.concat([training_data, test_data], axis=0)
    labels = cdf['fraud_label'].values

    # Drop non-feature columns
    cdf.drop(['fraud_label', 'index'], inplace=True, axis=1)

    # Create index of node features
    cdf.reset_index(inplace=True)
    meta_cols = ['client_node', 'merchant_node']
    for col in meta_cols:
        cdf[col] = cf.CategoricalIndex(cdf[col]).codes

    train_data, test_data, train_index, test_index, all_data = (cdf.iloc[:train_size, :],
                                                                cdf.iloc[train_size:, :],
                                                                cdf['index'][:train_size],
                                                                cdf['index'][train_size:],
                                                                cdf)
    return train_data, test_data, train_index, test_index, labels, all_data
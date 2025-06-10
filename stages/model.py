import os
import pickle
import typing

import cupy
import dgl
import torch
from dgl import nn as dglnn
from torch import nn
from torch.nn import functional as F

import cudf as cf


class BaseHeteroGraph(nn.Module):
    """
    Base class for Heterogeneous Graph Neural Network (GNN) models.
    """

    def __init__(self,
                 input_graph: dgl.DGLHeteroGraph,
                 embedding_size: int,
                 target: str):

        super().__init__()
        self._target = target

        # categorical embedding for all non-target node types
        self.hetero_embedding = dglnn.HeteroEmbedding(
            {ntype: input_graph.number_of_nodes(ntype)
             for ntype in input_graph.ntypes if ntype != self._target},
            embedding_size
        )

        self.layers = nn.ModuleList()

    def forward(self,
                graph_or_blocks: typing.Union[dgl.DGLHeteroGraph,
                                              typing.List[dgl.DGLHeteroGraph]],
                features: torch.Tensor
                ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        graph_or_blocks: full graph (inference) or list of sampled blocks (train/val)
        features: feature tensor for the target node type
        """

        # pick first block or full graph for embedding lookup
        blocks = graph_or_blocks
        first = blocks[0] if isinstance(blocks, list) else blocks

        # get initial embeddings for non-target types
        h_dict = self.hetero_embedding({
            ntype: first.nodes[ntype].data[dgl.NID]
            for ntype in self.hetero_embedding.embeds.keys()
        })

        # assign raw features for target node type
        h_dict[self._target] = features

        # apply every layer except the final linear
        for layer in self.layers[:-1]:
            h_dict = {k: F.leaky_relu(v) for k, v in h_dict.items()}
            h_dict = layer(blocks, h_dict)

        # final layer: linear over target embedding
        embedding = h_dict[self._target]
        out = self.layers[-1](embedding)
        return out, embedding

    def infer(self,
              graph: dgl.DGLHeteroGraph,
              features: torch.Tensor
              ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Full-graph inference: returns sigmoid scores and node embeddings.
        """
        logits, embed = self.forward(graph, features)
        return torch.sigmoid(logits), embed

    @torch.no_grad()
    def evaluate(self,
                 loader: dgl.dataloading.DataLoader,
                 feature_tensors: torch.Tensor,
                 target_node: str
                 ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate via neighbor-sampling DataLoader.
        Returns (logits, seeds, embeddings) concatenated over batches.
        """
        self.eval()
        all_logits, all_seeds, all_embeds = [], [], []

        for _, output_nodes, blocks in loader:
            seeds = output_nodes[target_node]
            nid = blocks[0].srcnodes[target_node].data[dgl.NID]
            feats = feature_tensors[nid]
            logits, embed = self.infer(blocks, feats)
            all_logits.append(logits.cpu())
            all_seeds.append(seeds.cpu())
            all_embeds.append(embed.cpu())

        return (
            torch.cat(all_logits, dim=0),
            torch.cat(all_seeds, dim=0),
            torch.cat(all_embeds, dim=0)
        )

    def inference(self,
                  graph: dgl.DGLHeteroGraph,
                  feature_tensors: torch.Tensor,
                  test_idx: torch.Tensor,
                  target_node: str = "transaction",
                  batch_size: int = 100
                  ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper: runs evaluate() over full-neighbor sampler.
        """
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=3)
        loader = dgl.dataloading.DataLoader(
            graph,
            {target_node: test_idx},
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        _, seeds, embeds = self.evaluate(loader, feature_tensors, target_node)
        return embeds, seeds


class HeteroRGCN(BaseHeteroGraph):
    """
    Heterogeneous Relational GCN.
    """

    def __init__(self,
                 input_graph: dgl.DGLHeteroGraph,
                 in_size: int,
                 hidden_size: int,
                 out_size: int,
                 n_layers: int,
                 embedding_size: int,
                 target: str = 'transaction'):

        super().__init__(input_graph, embedding_size, target)

        # determine input dims per relation
        in_dims = {
            rel: (in_size if src_type == self._target else embedding_size)
            for src_type, rel, _ in input_graph.canonical_etypes
        }

        # first layer
        self.layers.append(
            dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_dims[rel], hidden_size)
                for rel in input_graph.etypes
            }, aggregate='sum')
        )

        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(
                dglnn.HeteroGraphConv({
                    rel: dglnn.GraphConv(hidden_size, hidden_size)
                    for rel in input_graph.etypes
                }, aggregate='sum')
            )

        # final linear head
        self.layers.append(nn.Linear(hidden_size, out_size))


class HinSAGE(BaseHeteroGraph):
    """
    Heterogeneous GraphSAGE (HinSAGE).
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

        super().__init__(input_graph, embedding_size, target)

        # feature dims per node type
        in_feats = {
            ntype: (in_size if ntype == self._target else embedding_size)
            for ntype in input_graph.ntypes
        }

        # first HinSAGE conv
        convs = {}
        for src, rel, dst in input_graph.canonical_etypes:
            convs[rel] = dglnn.SAGEConv(
                (in_feats[src], in_feats[dst]),
                hidden_size,
                aggregator_type=aggregator_type
            )
        self.layers.append(dglnn.HeteroGraphConv(convs, aggregate='sum'))

        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(
                dglnn.HeteroGraphConv({
                    rel: dglnn.SAGEConv(hidden_size, hidden_size, aggregator_type=aggregator_type)
                    for rel in input_graph.etypes
                }, aggregate='sum')
            )

        # final linear head
        self.layers.append(nn.Linear(hidden_size, out_size))


def load_model(model_dir: str,
               gnn_model: typing.Type[BaseHeteroGraph] = HinSAGE,
               device: torch.device = None
               ) -> typing.Tuple[BaseHeteroGraph,
                                dgl.DGLHeteroGraph,
                                dict]:
    """
    Load graph.pkl, hyperparams.pkl, and model.pt from disk.
    """
    with open(os.path.join(model_dir, "graph.pkl"), 'rb') as f:
        graph = pickle.load(f)
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'rb') as f:
        hyperparameters = pickle.load(f)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = gnn_model(
        graph,
        in_size=hyperparameters['in_size'],
        hidden_size=hyperparameters['hidden_size'],
        out_size=hyperparameters['out_size'],
        n_layers=hyperparameters['n_layers'],
        embedding_size=hyperparameters['embedding_size'],
        target=hyperparameters['target_node']
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
    model.eval()
    return model, graph, hyperparameters


def build_fsi_graph(
    train_data: cf.DataFrame,
    col_drop: list[str]
) -> typing.Tuple[dgl.DGLHeteroGraph, torch.Tensor]:
    """
    Build a heterogeneous DGL graph and normalized feature tensor from cudf DataFrame.
    """
    # features: drop meta cols, convert to cupy, then to torch
    features = train_data.drop(columns=col_drop).values
    features = cupy.asarray(features)
    tensor = torch.from_dlpack(features.toDlpack())
    tensor = (tensor - tensor.mean(0, keepdim=True)) / (0.0001 + tensor.std(0, keepdim=True))

    # extract id tensors via cudf -> cupy -> torch
    ids = torch.tensor_split(
        torch.from_dlpack(train_data[col_drop].values.toDlpack()).long(),
        3, dim=1
    )
    client_tensor, merchant_tensor, transaction_tensor = (i.view(-1) for i in ids)

    edge_list = {
        ('client',      'buy',       'transaction'): (client_tensor,      transaction_tensor),
        ('transaction', 'bought',    'client'):      (transaction_tensor, client_tensor),
        ('transaction', 'issued',    'merchant'):    (transaction_tensor, merchant_tensor),
        ('merchant',    'sell',       'transaction'): (merchant_tensor,     transaction_tensor)
    }
    graph = dgl.heterograph(edge_list)
    return graph, tensor


def prepare_data(
    training_data: cf.DataFrame,
    test_data: cf.DataFrame
) -> typing.Tuple[cf.DataFrame, cf.DataFrame, cf.Series, cf.Series, cupy.ndarray, cf.DataFrame]:
    """
    Combine train/test cudf DataFrames, extract labels and node indices.
    Returns (train_df, test_df, train_idx, test_idx, labels, combined_df).
    """
    train_size = len(training_data)
    combined = cf.concat([training_data, test_data], axis=0)
    labels = combined['fraud_label'].values

    # drop non-feature columns
    combined = combined.drop(columns=['fraud_label', 'index'], errors='ignore')
    combined = combined.reset_index(drop=False)  # creates new 'index' column

    # encode meta columns
    for col in ['client_node', 'merchant_node']:
        combined[col] = cf.CategoricalIndex(combined[col]).codes

    train_df = combined.iloc[:train_size]
    test_df  = combined.iloc[train_size:]
    train_idx = train_df['index']
    test_idx  = test_df['index']

    return train_df, test_df, train_idx, test_idx, labels, combined
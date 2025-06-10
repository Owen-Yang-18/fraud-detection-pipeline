import os
import click                                           # Click for CLI handling :contentReference[oaicite:0]{index=0}
import cudf                                           # cuDF for fast CSV I/O 

from stages.graph_construction_stage import construct_fraud_graph
from stages.graph_sage_stage           import GraphSAGEStage
from stages.classification_stage       import ClassificationStage

@click.command()
@click.option('--training_file',  required=True, type=click.Path(exists=True),
              help='CSV of training transactions (with fraud_label).')
@click.option('--input_file',     required=True, type=click.Path(exists=True),
              help='CSV of transactions to score (same schema minus fraud_label).')
@click.option('--model_dir',      required=True, type=click.Path(exists=True, file_okay=False),
              help='Directory containing graph.pkl, hyperparams.pkl, model.pt, xgb.pt.')
@click.option('--output_file',    default='predictions.csv', show_default=True,
              help='Where to write the final fraud‐score CSV.')
@click.option('--batch_size',     default=100, show_default=True,
              help='Batch size for GNN inference.')
def main(training_file, input_file, model_dir, output_file, batch_size):
    """
    Locally‐adapted GNN fraud‐detection pipeline:
      1. Build the DGL heterograph and node features.
      2. Run GraphSAGE to append inductive embeddings.
      3. Classify (via GNN softmax or XGBoost) and write scores.
    """
    # 1) Graph construction
    graph, features, test_idx = construct_fraud_graph(training_file,
                                                     input_file)

    # 2) GNN embedding via GraphSAGEStage
    df_infer = cudf.read_csv(input_file)
    sage = GraphSAGEStage(model_dir=model_dir,
                          batch_size=batch_size)
    df_emb  = sage.run(df_infer, graph, features, test_idx)

    # (Optionally, you can inspect df_emb here before final classification.)

    # 3) Final scoring & output
    clf = ClassificationStage(model_dir=model_dir,
                              batch_size=batch_size)
    clf.run(training_file, input_file, output_file)

    click.echo(f"➜ Fraud scores written to {output_file}")

if __name__ == '__main__':
    main()

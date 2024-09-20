import argparse

argument_parser = argparse.ArgumentParser()

argument_parser.add_argument(
    "--arch", type=str, default="gcn",
    choices=("gcn", "sageconv", "attention", "leconv", "ginconv"),
    help="GNN Architecutures",
)

argument_parser.add_argument(
    "--node_feature", type=str, default="eigen",
    choices=("eigen", "identity", "connection", "degree"),
    help="Type of node feature to generate",
)

argument_parser.add_argument(
    "--threshold", type=float, default=2,
    choices=(0.5, 1, 1.5, 2, 2.5, 3), 
    help="Threshold for z scores of matrices"
)


# Architecture hyperparams --------------------------------

argument_parser.add_argument(
    "--num_layers", type=int, default=3, 
    choices=(1, 2, 3),
    help="Number of GNN layers"
)

argument_parser.add_argument(
    "--dropout", type=float, default=0.1,
    choices=(0.1, 0.05),
    help="Dropout rate for GNN layers"
)

argument_parser.add_argument(
    "--nhid", type=int, default=32,
    choices=(8, 16, 32),
    help="nhid"
)
# Training hyperparams --------------------------------


argument_parser.add_argument(
    "--lr_gnn", default=0.01, type=float,
    choices=(0.01, 0.001, 0.0001),
    help = "Learning rate of the GNN"
)

# Training PBS ID

argument_parser.add_argument(
    "--PBS_ID", default=1, type=int,
    help = "PBS ID Please")

# MLP Hyperparameters

argument_parser.add_argument(
    "--MLP_hid", default=16, type=int,
    choices=(16, 32, 64, 128),
    help = "hidden nodes of MLP"
)

args = argument_parser.parse_args()


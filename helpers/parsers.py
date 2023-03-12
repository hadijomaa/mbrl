import argparse
from loaders import SEARCH_SPACE_IDS


def get_hp_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    parser.add_argument('--search_space', type=str, choices=SEARCH_SPACE_IDS, default='4796',
                        help="Search space ID that represents the model class of interest")
    parser.add_argument('--data_directory', type=str, default="./hpob", help="directory of the pre-processed HP tasks")
    return parser


def get_transformer_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=1,
                        help="Number of layers in the encoder and decoder modules of the transformer")
    parser.add_argument('--num_heads', type=int, default=2,
                        help="Number of heads. Note that the number will be calculated to the power of 2")
    parser.add_argument('--d_model', type=int, default=8,
                        help="Dimensionality of the latent embedding."
                             "Note that the number will be calculated to the power of 2")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--dff', type=int, default=8,
                        help="Number of units used in the hidden layer of the FeedForward network"
                             "Note that the number will be calculated to the power of 2")
    parser.add_argument('--apply_scheduler', type=str, default="None", choices=["polynomial", "cosine", "None"],
                        help='Type of optimizer scheduler')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for inner updates')
    parser.add_argument('--meta_learning_rate', type=float, default=1e-3, help='learning rate for meta-updates')
    parser.add_argument('--meta_optimizer', type=str, default="sgd", choices=["adam", "radam", "sgd"],
                        help='Meta-optimizer')
    parser.add_argument('--optimizer', type=str, default="sgd", choices=["adam", "sgd"],
                        help='optimizer')
    return parser


def get_runner_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=0, help="Batch size when querying instances from tasks")
    parser.add_argument('--meta_batch_size', type=int, default=8,
                        help="Meta-batch size when querying tasks for meta-training")
    parser.add_argument('--seed', help='Seed', type=int, default=0)
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='Number of inner steps for the first order meta-learning')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default="./checkpoints", help='Checkpoint base directory')
    return parser

from helpers import parsers
from runners import Runner

if __name__ == "__main__":
    parser = parsers.get_runner_parser()
    parser = parsers.get_hp_parser(parser=parser)
    parser = parsers.get_transformer_parser(parser=parser)

    args = parser.parse_args()

    args.num_heads = 2 ** args.num_heads
    args.d_model = 2 ** args.d_model
    args.dff = 2 ** args.dff

    runner = Runner(args)

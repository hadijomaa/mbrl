from loaders.HPGenerator import MetaTaskGenerator
from modules.transformers import Transformer
import losses
from callbacks.reptile import ReptileCallback


class Runner:

    def __init__(self, args):
        self.data_directory = args.data_directory
        self.search_space = args.search_space

        self.seed = args.seed
        self.batch_size = args.batch_size
        self.meta_batch_size = args.meta_batch_size
        self.inner_steps = args.inner_steps
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.d_model = args.d_model
        self.dff = args.dff

        self.generator = MetaTaskGenerator(data_directory=self.data_directory, search_space_id=self.search_space,
                                           seed=self.seed, batch_size=self.batch_size,
                                           meta_batch_size=self.meta_batch_size, shuffle=True,
                                           inner_steps=self.inner_steps)
        self.model = Transformer(num_layers=self.num_layers, num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                                 dff=self.dff, d_model=self.d_model, num_latent=1)
        self.model.compile(loss=losses.nll, optimizer="adam", metrics=[losses.log_var, losses.mse])
        self.model.fit(self.generator)

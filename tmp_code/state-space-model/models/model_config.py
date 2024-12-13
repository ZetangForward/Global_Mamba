import argparse
from datetime import datetime
from functools import partial
from typing import List, Tuple, Union

from pydantic import BaseModel

from typing import Callable, Union


def import_from_str(path) -> Union[type, Callable]:
    """
    Imports a class or function from a string. 
    Format: "module.submodule.ClassName" or "module.submodule.function_name".

    Args: 
        path (str): The string representing the class or function to import.
    """
    module_name, obj_name = path.rsplit(".", 1)
    try:
        module = __import__(module_name, fromlist=[obj_name])
        obj = getattr(module, obj_name)
        return obj
    except ImportError as e:
        raise ImportError(f"Module '{module_name}' not found.") from e
    except AttributeError as e:
        raise AttributeError(f"Class '{obj_name}' not found in module '{module_name}'.") from e

class BaseConfig(BaseModel):
    @classmethod
    def from_cli(cls):
        import yaml
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--config', type=str, default=None, help='Path to the config file')
        parser.add_argument('--run_id', type=str, default=None, help='Run ID for the training')
        args, extra_args = parser.parse_known_args()


        if args.config is not None:
            with open(args.config) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            config = {}
        
        # Override with any extra arguments from the command line
        def _nested_update(config, args):
            for key, value in args.items():
                keys = key.split(".")
                for key in keys[:-1]:
                    config = config.setdefault(key, {})
                config[keys[-1]] = value

        extra_args = dict([arg.lstrip("-").split("=") for arg in extra_args])
        extra_args = {k.replace("-", "_"): v for k, v in extra_args.items()}
        _nested_update(config, extra_args)
        config = cls.parse_obj(config)

        if config.run_id is None:
            config.run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return config

    def print(self):
        try:
            import rich
            rich.print(self)
        except ImportError:
            print(self)


class FunctionConfig(BaseConfig):
    name: str
    kwargs: dict = {}

    def instantiate(self):
        return partial(import_from_str(self.name), **self.kwargs)

class ModuleConfig(BaseConfig):
    name: str
    kwargs: dict = {}

    def instantiate(self, **kwargs):
        return import_from_str(self.name)(**kwargs, **self.kwargs)

class DataSegmentConfig(BaseConfig):
    """
    This class should be subclassed to define per task. For example, MQARConfig
    """
    vocab_size: int = 8_192
    num_examples: int = 1_000
    input_seq_len: int = 64

    def build(self, **kwargs):
        raise NotImplementedError()
    
class DataConfig(BaseConfig):
    train_configs: List[DataSegmentConfig]
    test_configs: List[DataSegmentConfig]

    # can pass a tuple if you want a different batch size for train and test
    batch_size: Union[int, Tuple[int, int]] = 32

    seed: int = 123

    cache_dir: str = None
    force_cache: bool = False 

class ModelConfig(BaseConfig):
    sequence_mixer: ModuleConfig = None
    state_mixer: ModuleConfig = ModuleConfig(
        name="models.mixers.mlp.MLP", 
        kwargs={"hidden_mult": 2}
    )

    d_model: int = 768
    n_layers: int = 6
    max_position_embeddings: int = 1024
    learnable_word_embeddings: bool = True
    vocab_size: int = 50277

    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    drop_path: float = 0.0
    layer_norm_epsilon: float = 1e-5
    pad_vocab_size_multiple: int = 8

    block_type: str = "TransformerBlock"
    name: str = "default"

class LoggerConfig(BaseConfig):

    project_name: str = None
    entity: str = None
    

class TrainConfig(BaseConfig):
    data: DataConfig
    model: ModelConfig
    logger: LoggerConfig = LoggerConfig()

    max_epochs: int = 100

    # stop training once this metric reaches the threshold
    # set metric to None to disable early stopping
    early_stopping_metric: str = "valid/accuracy"
    early_stopping_threshold: float = 0.99
    slice_keys: List[str] = []

    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 123

    launch_id: str = None
    sweep_id: str = None
    run_id: str = "default"

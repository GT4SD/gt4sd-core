from gt4sd.frameworks.granular.train.core import (
    parse_arguments_from_config,
    train_granular,
)

args = parse_arguments_from_config("config_ae.ini")

train_granular(vars(args))

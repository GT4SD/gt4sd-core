#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import argparse
import configparser
from typing import Any, Dict, Optional

import sentencepiece as _sentencepiece
import numpy as np
from pytorch_lightning import Trainer

from ..ml.models import MODEL_FACTORY
from .utils import convert_string_to_class

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece


def parse_arguments_from_config(conf_file: Optional[str] = None) -> argparse.Namespace:
    """Parse arguments from configuration file.

    Args:
        conf_file: configuration file. Defaults to None, a.k.a. us a default configuration
            in ./config/config.ini.

    Returns:
        the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # open config.ini file, either from parser or default file
    parser.add_argument(
        "--conf_file",
        type=str,
        help=("config file for the defaults value"),
        default="./config/config.ini",
    )

    # Read config file
    args, remaining_argv = parser.parse_known_args()
    config = configparser.ConfigParser()

    if conf_file:
        config.read(conf_file)
    else:
        config.read(args.conf_file)

    # classes that are not model name
    general_config_classes = ["general", "trainer", "default"]

    # adding a list of all model name into the args
    result: Dict[str, Any] = dict()
    result["model_list"] = [
        i for i in list(config.keys()) if i.lower() not in general_config_classes
    ]
    for key in [*config.keys()]:
        # go trough all models parameter, replace the parsed ones from the the config files ones
        if key.lower() not in general_config_classes:
            model_type = config[key]["type"]
            params_from_configfile = dict(config[key])
            model = MODEL_FACTORY[model_type.lower()]
            parser = model.add_model_specific_args(parser, key)
            args, _ = parser.parse_known_args()
            args_dictionary = vars(args)
            params_from_configfile["name"] = key

            for i in params_from_configfile:
                params_from_configfile[i] = convert_string_to_class(
                    params_from_configfile[i]
                )

            params_from_configfile.update(
                {
                    k[: -len(key) - 1]: v
                    for k, v in args_dictionary.items()
                    if v is not None and k.endswith("_" + key)
                }
            )

            result[key] = params_from_configfile

        elif key.lower() == "trainer" or key.lower() == "general":
            params_from_configfile = dict(config[key])
            for i in params_from_configfile:
                params_from_configfile[i] = convert_string_to_class(
                    params_from_configfile[i]
                )
            result.update(params_from_configfile)

    # parser Pytorch Trainer arguments
    parser = Trainer.add_argparse_args(parser)

    # parse the arguments
    parser.add_argument("--algorithm", type=str, default="trajectory_balance")
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--dataset_path", type=str, default="./data/qm9.h5")
    parser.add_argument("--environment", type=str, default=None)
    parser.add_argument("--model", type=str, default="graph_transformer_gfn")
    parser.add_argument("--sampling_model", type=str, default="graph_transformer_gfn")
    parser.add_argument("--task", type=str, default="qm9")

    # adding basename as the name of the run
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--basename", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--validation_split", type=float)
    parser.add_argument("--validation_indices_file", type=str)
    parser.add_argument("--stratified_batch_file", type=str)
    parser.add_argument("--stratified_value_name", type=str)
    parser.add_argument("--checkpoint_every_n_val_epochs", type=str)
    parser.add_argument("--trainer_log_every_n_steps", type=str)
    parser.add_argument("--trainer_flush_logs_every_n_steps", type=str)
    parser.add_argument("--test_output_path", type=str, default="./test")

    parser.add_argument("--log_dir", type=str, default="./log/")
    parser.add_argument("--num_training_steps", type=int, default=1000)
    parser.add_argument("--validate_every", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=142857)
    parser.add_argument("--device", type=str, default="cpu")

    # arguments specific for gflownet
    parser.add_argument("--bootstrap_own_reward", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--global_batch_size", type=int, default=16)
    parser.add_argument("--num_emb", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--tb_epsilon", type=float, default=1e-10)
    parser.add_argument("--illegal_action_logreward", type=float, default=-50)
    parser.add_argument("--reward_loss_multiplier", type=float, default=1)
    parser.add_argument("--temperature_sample_dist", type=str, default="uniform")
    parser.add_argument("--temperature_dist_params", type=str, default="(.5, 32)")
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--num_data_loader_workers", type=float, default=8)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--lr_decay", type=int, default=20000)
    parser.add_argument("--z_lr_decay", type=int, default=20000)
    parser.add_argument("--clip_grad_type", type=str, default="norm")
    parser.add_argument("--clip_grad_param", type=int, default=10)
    parser.add_argument("--random_action_prob", type=float, default=0.001)
    parser.add_argument("--sampling_tau", type=float, default=0.0)
    parser.add_argument("--max_nodes", type=int, default=9)
    parser.add_argument("--num_offline", type=int, default=10)
    parser.add_argument("--sampling_iterator", type=bool, default=True)
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--development_mode", type=bool, default=False)

    args_dictionary = vars(parser.parse_args(remaining_argv))
    result.update({k: v for k, v in args_dictionary.items() if v is not None})
    # add random number generator from seed
    result["rng"] = np.random.default_rng(result["seed"])
    result_namespace = argparse.Namespace(**result)

    return result_namespace

import torch
import pandas as pd
import os
import numpy as np

from gt4sd.frameworks.granular.train.core import parse_arguments_from_config, train_granular
from gt4sd.frameworks.granular.ml.module import GranularModule
from gt4sd.frameworks.granular.dataloader.dataset import build_dataset_and_architecture
from gt4sd.frameworks.granular.dataloader.data_module import GranularDataModule


def create_z0(model_combiner, datasets, batch_size=500, pickel_z0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    dm = GranularDataModule(datasets,
                           batch_size=batch_size,
                           validation_split=0.5,
                           validation_indices_file=None,
                           num_workers=1)

    dm.prepare_test_data(datasets)
    dm.setup(stage='test')

    z0 = {}
    mems_0 = {}
    y_true = {}
    with torch.no_grad():
        for auto_enco in model_combiner.autoencoders:
            z_temp = []
            mems_temp = []
            auto_enco.to(device)
            auto_enco.eval()
            for i, batch in enumerate(dm.train_dataloader()):
                x_out = auto_enco.encoder(batch[auto_enco.input_key].to(device))
                z_ = auto_enco.encode(batch[auto_enco.input_key].to(device))
                z_temp.append(z_)
            z0[auto_enco.position] = torch.cat(z_temp, dim=0).detach().cpu().numpy()

        y_pred = {}
        for model in model_combiner.latent_models:
            z_model_input = torch.cat([torch.tensor(z0[pos]) for pos in model.from_position], dim=1)
            z_model_input = z_model_input.to(device)
            model.to(device)
            y_pred[model.name] = model.predict(x=z_model_input).detach().cpu().numpy()
            y_true_temp = []
            for i, batch in enumerate(dm.train_dataloader()):
                y_true_temp.append(batch[model.target_key])
            torch.cuda.empty_cache()
            y_true[model.name] = torch.cat(y_true_temp, dim=0).detach().cpu().numpy()
            #print(y_pred[model.name].shape)

    pd.to_pickle(z0, pickel_z0+'_z0_pickle.pkl', protocol=4)
    pd.to_pickle(y_pred,  pickel_z0+'_y_pred.pkl', protocol=4)

    col_names = []
    for auto_enco in model_combiner.autoencoders:
        col_names.extend([auto_enco.name + str(i) for i in range(z0[auto_enco.position].shape[1])])
    for latent_models in model_combiner.latent_models:
        col_names.extend([latent_models.name + str(i) for i in range(y_pred[latent_models.name].shape[1])])

    np_res_temp = []
    for auto_enco in model_combiner.autoencoders:
        np_res_temp.append(z0[auto_enco.position])
    for latent_models in model_combiner.latent_models:
        np_res_temp.append(y_true[latent_models.name])
    np_res = np.concatenate(np_res_temp, axis=1)

    df = pd.DataFrame(np_res, columns=col_names)
    df.to_pickle(pickel_z0+'_z0_df.pkl', protocol=4)
    return df


def create_z_df_from_config(conf_file, model_combiner):
    args = parse_arguments_from_config(conf_file)
    hparams_args = vars(args)

    datasets = []
    for model in hparams_args['model_list']:
        #print('Dataset preparation for: ', model)

        hparams_model = hparams_args[model]
        model_type = hparams_model['type'].lower()

        dataset, _ = build_dataset_and_architecture(hparams_model['name'],
                                                    hparams_model['data_path'],
                                                    hparams_model['data_file'],
                                                    hparams_model['dataset_type'],
                                                    hparams_model['type'],
                                                    hparams_model)
        datasets.append(dataset)

    z0_df = create_z0(model_combiner, datasets, pickel_z0='my_prefix')
    return z0_df


if __name__ == "__main__":

    model = GranularModule.load_from_checkpoint('logs/<path_to_/../checkpoints>/epoch=....ckpt')
    model.eval()
    create_z_df_from_config('config_ae.ini', model)


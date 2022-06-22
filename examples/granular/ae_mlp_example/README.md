# Granular: train / test for AE+MLP

An example to train and test AE with MLP property predictors.

Make sure to activate the conda environment:

```console
conda activate gt4sd
```

We assume you have in the working directory a valid config .ini called `config_ae.ini`, more details in the training [script](./example_train_ae_and_mlp.py).
Train the models with:

```console
python example_train_ae_and_mlp.py
```

Use the checkpoint generated for testing and saving the latent space:

```console
python example_test_ae_and_mlp.py
```

*NOTE:* update the path to the checkpoint in the test [script](./example_test_ae_and_mlp.py).


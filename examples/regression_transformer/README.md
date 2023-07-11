# Regression Transformer

A simple example to train, save and run inference on a custom Regression Transformer (RT) model.

First, activate the conda environment:

```console
conda activate gt4sd
```

## Training a model
We generally recommend to finetune an existing RT model.
### Finetuning 
To launch a finetuning of a RT pretrained on drug-like molecules from ChEMBL, execute the following from the GT4SD root:

```console
 gt4sd-trainer --training_pipeline_name regression-transformer-trainer --model_path ~/.gt4sd/algorithms/conditional_generation/RegressionTransformer/RegressionTransformerMolecules/qed --do_train --output_dir my_regression_transformer --train_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --test_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --overwrite_output_dir --eval_steps 200 --augment 10 --eval_accumulation_steps 1 --num_train_epochs 100 
```
*NOTE*: This is *dummy* example, do not use "as is" :warning: Substitute the train/test data path to point to your files. You can inspect the format of the example file to see the needed format (`.csv` with the first column called "text" and one or multiple property columns). Adjust the remaining arguments as desired. See full API [here](https://gt4sd.github.io/gt4sd-core/api/gt4sd.training_pipelines.regression_transformer.core.html).

*NOTE*: Substitute the `--model_path` to the directory of the pretrained model. GT4SD provides RT models pretrained on:
- **small molecules**: 
  - Models for single properties: `qed`, `esol`, `crippen_logp`
  - Models for multiple properties: `logp_and_synthesizability`, `cosmo_acdl`, `pfas`
- **proteins**: `stability`
- **chemical reactions**: `uspto`
- **polymers**: `rop_catalyst` and `block_copolymer`, both described in [Park et al., (2023; Nature Communications)](https://www.nature.com/articles/s41467-023-39396-3)
For details on these model versions, see the [HF spaces app](https://huggingface.co/spaces/GT4SD/regression_transformer).

*NOTE*: :warning: The above assumes that you have the `qed` model cached locally. If this is not the case, run an inference to trigger the caching mechanism:

```py
from gt4sd.algorithms.registry import ApplicationsRegistry
algorithm = ApplicationsRegistry.get_application_instance(
  target='CCO',
  sampling_wrapper={'property_goal': {'<qed>': 0.12}},
  algorithm_type='conditional_generation',
  domain='materials',
  algorithm_name='RegressionTransformer',
  algorithm_application='RegressionTransformerMolecules',
  algorithm_version='qed'
)
```
Consider replacing `algorithm_version='qed'`, dependent on which model you want to finetune.

### Training from scratch
We generally recommend finetuning, but if you want to train a model from scratch, just remove the `--model_path` arg and instead use `--config_name` to point to your custom XLNet/RT configuration file and use `--tokenizer_name` to point to an existing HF tokenizer or to a custom path. E.g.:

```console
gt4sd-trainer --training_pipeline_name regression-transformer-trainer --tokenizer_name ~/.gt4sd/algorithms/conditional_generation/RegressionTransformer/RegressionTransformerMolecules/qed --config_name examples/regression_transformer/rt_config.json  --do_train --output_dir my_regression_transformer --train_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --test_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --overwrite_output_dir --eval_steps 200 --augment 5 --eval_accumulation_steps 1
```
For convenience, this command still uses a tokenizer from an existing model. New tokens contained in your dataset will be added automatically.

## Saving 
After completing the training, *save* the RT model in a gt4sd-compatible manner. This step is necessary to later run `gt4sd-inference` on it, just like you can do it with any native/pretrained RT model.

```console
gt4sd-saving --training_pipeline_name regression-transformer-trainer --model_path my_regression_transformer --target_version fast-example-v0 --algorithm_application RegressionTransformerMolecules
```

This simply moves the model artifacts to the local GT4SD cache. Using *as is*, this command uses the files from the last checkpoint, saved simply as files inside `my_regression_transformer`. If you want to save files of a specific checkpoint, just use `--checkpoint_name my-desired-checkpoint`.

## Inference
Now you can run inference on your custom model. 

#### Predict
To predict the properties of a molecule, provide a masked query:

```console
gt4sd-inference --algorithm_name RegressionTransformer --algorithm_application RegressionTransformerMolecules --target "<prop0>[MASK][MASK][MASK][MASK][MASK][MASK][MASK]|<prop1>[MASK][MASK][MASK][MASK][MASK][MASK]|<prop2>[MASK][MASK][MASK][MASK][MASK]|<prop3>[MASK][MASK][MASK][MASK][MASK][MASK]|<prop4>[MASK][MASK][MASK][MASK][MASK]|<prop5>[MASK][MASK][MASK][MASK][MASK]|<prop6>[MASK][MASK][MASK][MASK][MASK]|<prop7>[MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK]|[C][C][C][O][C][Branch1_2][C][=O][C][Branch2_1][Ring1][N][C][=C][C][=C][C][=C][Ring1][Branch1_2][C][Branch1_1][Branch2_2][C][=C][C][=C][C][=C][Ring1][Branch1_2][C][=C][C][=C][C][=C][Ring1][Branch1_2][C][Branch1_1][C][F][Branch1_1][C][F][S][Branch1_2][C][=O][Branch1_2][C][=O][O-expl].[C][C][O]" --algorithm_version fast-example-v0  --number_of_samples 1 --configuration_file examples/regression_transformer/inference_predict.json
```

#### Generate
To conditionally generate a molecule, provide a SMILES as target and specify the search constraints in `examples/regression_transformer/inference_predict.json`:

```console
gt4sd-inference --algorithm_name RegressionTransformer --algorithm_application RegressionTransformerMolecules --target "CCCOC(=O)C(C1=CC=CC=C1C(C2=CC=CC=C2)C3=CC=CC=C3)C(F)(F)S(=O)(=O)[O-].CCO" --algorithm_version fast-example-v0 --number_of_samples 10 --configuration_file examples/regression_transformer/inference_generate.json
```


## Sharing
Do not forget to upload your model to the GT4SD model hub. You can do this via:

```console
gt4sd-upload --training_pipeline_name regression-transformer-trainer --algorithm_application RegressionTransformerMolecules --model_path my_regression_transformer --target_version debugging-rt 
```

Afterwards everybody can sync your model from the hub and run it just like our other pretrained models.


### Citation
For details on this methodology see:

```bib
@article{born2023regression,
  title={Regression Transformer enables concurrent sequence regression and generation for molecular language modelling},
  author={Born, Jannis and Manica, Matteo},
  journal={Nature Machine Intelligence},
  year={2023},
  month={04},
  day={06},
  volume={5},
  number={4},
  pages={432--444},
  doi={10.1038/s42256-023-00639-z},
  url={https://doi.org/10.1038/s42256-023-00639-z},
}
```

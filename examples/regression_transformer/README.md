# Regression Transformer: 

A simple example to finetune the Regression Transformer (RT) model

Make sure to activate the conda environment:

```console
conda activate gt4sd
```

To launch a finetuning of a RT pretrained on drug-like moelcules from ChEMBL, execute the following from the GT4SD root:

```console
 gt4sd-trainer --training_pipeline_name regression-transformer-trainer --model_path ~/.gt4sd/algorithms/conditional_generation/RegressionTransformer/RegressionTransformerMolecules/qed --do_train --output_dir dummy_regression_transformer --train_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --test_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --overwrite_output_dir --eval_steps 2 --augment 10 --eval_accumulation_steps 1
```
*NOTE*: This is *dummy* example, do not use "as is" :warning:

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


After completing the training, *save* the RT model in a gt4sd-compatible manner (to later run `gt4sd-inference` on it):

```console
gt4sd-saving --training_pipeline_name regression-transformer-trainer --model_path dummy_regression_transformer --train_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --test_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --eval_steps 2 --augment 10 --eval_accumulation_steps 1 --num_train_epochs 10 --training_name fast-example --target_version fast-example-v0 --algorithm_application RegressionTransformerMolecules --checkpoint_name checkpoint-final
```

For details on this methodology see:

```bib
@article{born2022regression,
  title={Regression Transformer: Concurrent Conditional Generation and Regression by Blending Numerical and Textual Tokens},
  author={Born, Jannis and Manica, Matteo},
  journal={arXiv preprint arXiv:2202.01338},
  note={Spotlight talk at ICLR workshop on Machine Learning for Drug Discovery},
  year={2022}
}
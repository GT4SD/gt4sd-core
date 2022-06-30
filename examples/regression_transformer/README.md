# Regression Transformer: 

A simple example to finetune the Regression Transformer (RT) model

Make sure to activate the conda environment:

```console
conda activate gt4sd
```

To launch a finetuning of a RT pretrained on drug-like moelcules from ChEMBL, execute the following from the GT4SD root:

```console
 gt4sd-trainer --training_pipeline_name regression-transformer-trainer --model_path ~/.gt4sd/algorithms/conditional_generation/RegressionTransformer/RegressionTransformerMolecules/qed --do_train --output_dir dummy_regression_transformer --train_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --test_data_path src/gt4sd/training_pipelines/tests/regression_transformer_raw.csv --overwrite_output_dir --eval_steps 2 --augment 10 --test_fraction 0.2 --eval_accumulation_steps 1
```
*NOTE*: This is *dummy* example, do not use "as is" :warning:

For details on this methodology see:

```bib
@article{born2022regression,
  title={Regression Transformer: Concurrent Conditional Generation and Regression by Blending Numerical and Textual Tokens},
  author={Born, Jannis and Manica, Matteo},
  journal={arXiv preprint arXiv:2202.01338},
  note={Spotlight talk at ICLR workshop on Machine Learning for Drug Discovery},
  year={2022}
}
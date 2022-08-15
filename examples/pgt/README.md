# PGT

A simple example to train the PGT model

Make sure to activate the conda environment:

```console
conda activate gt4sd
```

To launch a training execute the following command from the GT4SD root. A dummy `patent_dataset.jsonl` is provided to check the needed input format. Replace it with an actual dataset for a valid training.

```console

gt4sd-trainer  --training_pipeline_name language-modeling-trainer \
    --model_name_or_path gpt2 \
    --lr 2e-5 \
    --batch_size 2 \
    --train_file patent_dataset.jsonl \
    --validation_file patent_dataset.jsonl \
    --type clm \
    --lr_decay 0.5 \
    --default_root_dir output_pgt \
    --max_epochs 3 \
    --val_check_interval 50000 \
    --limit_val_batches 500 \
    --accumulate_grad_batches 4 \
    --log_every_n_steps 500 \
    --monitor val_loss \
    --save_top_k 10 \
    --mode min \
    --every_n_train_steps 50000 \
    --accelerator 'ddp' 
```



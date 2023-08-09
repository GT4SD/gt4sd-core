# Molformer

Simple example to train or finetune the Molformer model

Make sure to activate the conda environment:

```console
conda activate gt4sd
```

## Pretraining

An example for Molformer's pretraining. The `data_path` parameter contains the path where one or both the `pubchem`, `ZINC` directories are located. Link to the dataset and further detais about it can be found at the [original molformer repo](https://github.com/IBM/molformer).

```console

gt4sd-trainer  --training_pipeline_name molformer \
        --type pretraining \
        --batch_size 1200  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 8\
        --max_epochs 4\
        --num_feats 32 \
        --grad_acc 1\
        --data_path molformer/data/pretrained \
        --model_arch BERT_both_rotate
```

## Finetuning 

### Classification

An example of classification finetuning using the hiv dataset. Link to the dataset can be found at the [original molformer repo](https://github.com/IBM/molformer).

```console

gt4sd-trainer  --training_pipeline_name molformer \
        --type classification \
        --batch_size 128  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 8\
        --max_epochs 500\
        --num_feats 32 \
        --every_n_epochs 10 \
        --data_root molformer/data/hiv \
        --pretrained_path pretrained_molformer/checkpoints/N-Step-Checkpoint_3_30000.ckpt \
        --dataset_name hiv \
        --measure_name HIV_active \
        --dims 768 768 768 1 \
        --num_classes 2 \
        --save_dir test_hiv 
```

### Multiclass classification

An example of multiclass finetuning using the clintox dataset. Link to the dataset can be found at the [original molformer repo](https://github.com/IBM/molformer).

```console

gt4sd-trainer --training_pipeline_name molformer \
        --type multitask_classification \
        --batch_size 128  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 8\
        --max_epochs 500\
        --num_feats 32 \
        --every_n_epochs 10 \
        --data_root molformer/data/clintox \
        --pretrained_path pretrained_molformer/checkpoints/N-Step-Checkpoint_3_30000.ckpt \
        --dataset_name tox21 \
        --dims 768 768 768 1 \
        --measure_names FDA_APPROVED CT_TOX
        --save_dir test_clintox \
```

### Regression

An example of regression finetuning using the qm9 dataset. Link to the dataset can be found at the [original molformer repo](https://github.com/IBM/molformer).

```console

gt4sd-trainer  --training_pipeline_name molformer \
        --type regression \
        --n_batch 128  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --n_workers 8\
        --max_epochs 500\
        --num_feats 32 \
        --every_n_epochs 10 \
        --data_root molformer/data/qm9 \
        --pretrained_path pretrained_molformer/checkpoints/N-Step-Checkpoint_3_30000.ckpt \
        --dataset_name qm9 \
        --measure_name alpha \
        --dims 768 768 768 1 \
        --save_dir test_alpha
```



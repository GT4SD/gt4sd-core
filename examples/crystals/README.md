## Crystals

A simple example to train CGCNN based models to predict material properties.

Make sure to activate the conda environment:

```console
conda activate gt4sd
```

To launch a training execute the following command from the GT4SD root. Examples of a dataset, including he sample-classification that we use in the given example, can be found in the CGCNN's official implementation repo in the following [link](https://github.com/txie-93/cgcnn/tree/master/data).

```console

gt4sd-trainer  --training_pipeline_name cgcnn \
    --task classification \
    --datapath sample-classification \
    --atom_fea_len 64 \
    --h_fea_len 128 \
    --n_conv 3 \
    --n_h 1 \
    --epochs 30 \
    --batch_size 256 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 0.0 \
    --optim SGD 
```

The code is adapted from: <https://github.com/txie-93/cgcnn>.  

```bibtex
@article{xie2018crystal,
  title={Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties},
  author={Xie, Tian and Grossman, Jeffrey C},
  journal={Physical review letters},
  volume={120},
  number={14},
  pages={145301},
  year={2018},
  publisher={APS}
}
```


Federated learning experiments

# Setup

This README assumes that TensorFlow Federated and all other requirements are appropriately configured. See the [instructions](https://github.com/tensorflow/federated/blob/master/docs/install.md) on the TFF website for more details.

# LARC dataset

This repo uses the University of Michigan [Learning Analytics Architecture](https://enrollment.umich.edu/data-research/learning-analytics-data-architecture-larc) sample dataset.

From the raw sample dataset tables, construct a single joined example dataset:

```
export LARC_MOCK="data/larc-mock/larc-mock.csv"

python scripts/generate_combined_larc_dataset.py \
    --prsn_identifying_info_fp data/larc-mock/PRSN_IDNTFYNG_INFO.csv \
    --stdnt_info_fp data/larc-mock/STDNT_INFO.csv \
    --stdnt_term_class_info_fp data/larc-mock/STDNT_TERM_CLASS_INFO.csv \
    --stdnt_term_info_fp data/larc-mock/STDNT_TERM_INFO.csv \
    --stdnt_term_trnsfr_info_fp data/larc-mock/STDNT_TERM_TRNSFR_INFO.csv \
    --out_fp $LARC_MOCK
```

This generates a single complete dataset called `larc-mock.csv` by joining the tables.

Next, we can train an example federated model from this dataset:

``` 
python train_feded_model.py --data_fp $LARC_MOCK
```
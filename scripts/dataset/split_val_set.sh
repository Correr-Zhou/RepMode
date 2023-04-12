declare -a arr=(
    "beta_actin"
    "myosin_iib"
    "membrane_caax_63x"
    "desmoplakin"
    "sec61_beta"
    "st6gal1"
    "fibrillarin"
    "lamin_b1"
    "alpha_tubulin"
    "tom20"
    "zo1"
)
for i in "${arr[@]}"
do
    python scripts/python/split_dataset_val.py \
        "data/csvs/${i}/train.csv" \
        "data/csvs" \
        --train_size 0.9 \
        -v
done

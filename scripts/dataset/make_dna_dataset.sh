python scripts/python/make_dataset.py \
    "data/csvs" \
    "data/csvs/dna.csv" \
    "data/csvs" \
    "train" \
    --sample_num 54 \
    -v \

python scripts/python/make_dataset.py \
    "data/csvs" \
    "data/csvs/dna.csv" \
    "data/csvs" \
    "val" \
    --sample_num 6 \
    -v \

python scripts/python/make_dataset.py \
    "data/csvs" \
    "data/csvs/dna.csv" \
    "data/csvs" \
    "test" \
    --sample_num 20 \
    -v


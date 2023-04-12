GPU_IDX=${1:-0}
MODEL_NAME=${2:-RepMode}
EXP_NAME=${3:-exps/test}

python main.py \
       --nn_module ${MODEL_NAME}  \
       --path_exp_dir ${EXP_NAME} \
       --gpu_ids ${GPU_IDX} \
       --path_load_dataset data/all_data \
       --num_epochs 1000 \
       --batch_size 8 \
       --lr 0.0001 \
       --interval_val 20 \
       # --save_test_preds \
       # --save_test_signals_and_targets \
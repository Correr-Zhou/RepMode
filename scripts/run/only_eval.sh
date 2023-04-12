GPU_IDX=${1:-0}
MODEL_NAME=${2:-RepMode}
EXP_NAME=${3:-exps/test}
MODEL_PATH=${4:-exps/test/checkpoints/model_best_test.p}

python eval.py \
       --nn_module ${MODEL_NAME} \
       --path_exp_dir ${EXP_NAME} \
       --path_load_model ${MODEL_PATH} \
       --gpu_ids ${GPU_IDX} \
       --path_load_dataset data/all_data \
       # --save_test_preds \
       # --save_test_signals_and_targets \

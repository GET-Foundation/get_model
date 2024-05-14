CUDA_VISIBLE_DEVICES=1 python /pmglocal/alb2281/repos/get_model/get_model/baselines/hyena/train_hyena.py \
    --data_dir /pmglocal/alb2281/repos/get_model/get_model/baselines/hyena/data \
    --batch_size 400 \
    --num_epochs 100 \
    --wandb_project_name get-cre-pred-hyena \
    --wandb_run_name cre_pred_hyena_10_balanced_cls \
    --split_data \
    --balanced_split
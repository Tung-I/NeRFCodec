cd ../..

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/jpeg65_tv1e5.txt \
                --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 15000 \
                --save_every 5000 --vis_every 5000 \
                --refresh_k 32 
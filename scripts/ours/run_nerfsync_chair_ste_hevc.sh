cd ../..
python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_hevc_qp22.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project nerf_chair_ste \
                --n_iters 20000 \
                --save_every 2000 --vis_every 2000

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_hevc_qp32.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project nerf_chair_ste \
                --n_iters 20000 \
                --save_every 2000 --vis_every 2000

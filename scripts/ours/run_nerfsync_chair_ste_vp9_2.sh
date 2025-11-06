cd ../..
python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_vp9_qp28.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ste_nerf_chair \
                --n_iters 20000 \
                --save_every 2000 --vis_every 2000 \
                --refresh_k 32

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_vp9_qp44.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ste_nerf_chair \
                --n_iters 20000 \
                --save_every 2000 --vis_every 2000 \
                --refresh_k 32